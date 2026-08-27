from __future__ import annotations

from dataclasses import dataclass

from prowl import ProwlResult
from src.agent.ollama import LLMResult, OllamaClient
from support import SupportCard

_DIAGNOSTIC_WORDS = frozenset(
    {"broken", "cannot", "crash", "error", "failed", "failure", "log", "logs", "stuck", "won't"}
)


@dataclass(frozen=True)
class RenderedAnswer:
    text: str
    route: str | None = None


def _is_diagnostic(query: str) -> bool:
    words = {word.strip(".,?!:;`()[]{}").lower() for word in query.split()}
    return bool(words & _DIAGNOSTIC_WORDS)


def _prompt(query: str, card: SupportCard, sources: ProwlResult | None) -> str:
    parts = [
        f"Question: {query}",
        f"Reviewed support answer: {card.answer}",
        f"Risk: {card.risk}",
    ]
    if card.docs_url:
        parts.append(f"Documentation: {card.docs_url}")
    if sources is not None and sources.status == "ok":
        parts.append("Indexed source locations:")
        parts.extend(f"- {hit.citation}: {hit.snippet}" for hit in sources.hits)
    parts.append("Reply with only a concise Discord-ready final answer. Do not add facts beyond this evidence.")
    return "\n".join(parts)


class Answerer:
    def __init__(self, client: OllamaClient):
        self.client = client

    async def render(
        self,
        query: str,
        card: SupportCard,
        sources: ProwlResult | None = None,
    ) -> RenderedAnswer:
        if card.risk == "destructive":
            return RenderedAnswer(card.answer)
        route = "lfm" if sources is not None and sources.status == "ok" and _is_diagnostic(query) else "gemma"
        result: LLMResult = await self.client.answer(_prompt(query, card, sources), route=route)
        if result.status != "ok":
            return RenderedAnswer(card.answer)
        return RenderedAnswer(result.text, route=route)
