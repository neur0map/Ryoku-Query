from __future__ import annotations

import asyncio
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Awaitable, Callable

_MAX_FINAL_CHARS = 1800
_THINKING_BLOCK = re.compile(r"<think>.*?</think>\s*", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class LLMResult:
    status: str
    text: str = ""
    error: str | None = None


Runner = Callable[[str, dict, float], Awaitable[dict]]


def _clean_final_answer(value: object) -> str:
    if not isinstance(value, str):
        return ""
    if re.search(r"<think\b", value, re.IGNORECASE) and not re.search(
        r"</think\s*>", value, re.IGNORECASE
    ):
        return ""
    return _THINKING_BLOCK.sub("", value).strip()[:_MAX_FINAL_CHARS]


def _post_json(url: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("Ollama returned an invalid response")
    return parsed


async def _request(url: str, payload: dict, timeout: float) -> dict:
    return await asyncio.to_thread(_post_json, url, payload, timeout)


class OllamaClient:
    def __init__(
        self,
        host: str,
        gemma_model: str,
        lfm_model: str,
        *,
        timeout: float = 45.0,
        runner: Runner | None = None,
    ):
        self.host = host.rstrip("/")
        self.gemma_model = gemma_model
        self.lfm_model = lfm_model
        self.timeout = min(max(float(timeout), 1.0), 180.0)
        self.runner = runner or _request
        self._request_lock = asyncio.Semaphore(1)

    async def answer(self, prompt: str, *, route: str) -> LLMResult:
        if route not in {"gemma", "lfm"}:
            raise ValueError("route must be gemma or lfm")
        payload = {
            "model": self.gemma_model if route == "gemma" else self.lfm_model,
            "stream": False,
            "think": route == "lfm",
            "keep_alive": "0",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are Ryoku Help in Discord. Answer only from the supplied reviewed support evidence "
                        "and indexed source evidence. Never invent commands, paths, URLs, system state, or citations. "
                        "Give a concise, plain-language final answer with one safe next step; ask one focused "
                        "clarifying question when the evidence does not identify the symptom. For a state-changing "
                        "action, name its effect and prefer a read-only check first unless the user explicitly asks "
                        "to perform it. For contributor/source questions, distinguish a packaged install from a dev "
                        "checkout when the evidence does. Never reveal private reasoning, chain of thought, or analysis."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "options": {"temperature": 0.1, "num_predict": 1200 if route == "lfm" else 280},
        }
        try:
            async with self._request_lock:
                response = await self.runner(
                    f"{self.host}/api/chat", payload, self.timeout
                )
                first_message = response.get("message") if isinstance(response, dict) else None
                first_text = _clean_final_answer(
                    first_message.get("content") if isinstance(first_message, dict) else None
                )
                if route == "lfm" and not first_text:
                    retry_payload = dict(payload)
                    retry_payload["think"] = False
                    retry_payload["options"] = dict(payload["options"])
                    retry_payload["options"]["num_predict"] = 400
                    response = await self.runner(
                        f"{self.host}/api/chat", retry_payload, self.timeout
                    )
        except (OSError, TimeoutError, urllib.error.URLError, json.JSONDecodeError) as error:
            return LLMResult("unavailable", error=f"Local model unavailable: {error}")
        message = response.get("message")
        if not isinstance(message, dict):
            return LLMResult("unavailable", error="Local model returned no message")
        text = _clean_final_answer(message.get("content"))
        if not text:
            return LLMResult("unavailable", error="Local model returned no final answer")
        return LLMResult("ok", text=text)
