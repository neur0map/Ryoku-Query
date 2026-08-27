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
        # This host cannot safely hold Gemma and LFM resident together.
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
                        "You are Ryoku Help in Discord. Answer only from the supplied "
                        "evidence. Give a short, direct final answer. Never reveal private "
                        "reasoning, chain of thought, or analysis."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "options": {"temperature": 0.1, "num_predict": 700 if route == "lfm" else 280},
        }
        try:
            async with self._request_lock:
                response = await self.runner(
                    f"{self.host}/api/chat", payload, self.timeout
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
