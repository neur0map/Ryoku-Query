from __future__ import annotations

from dataclasses import dataclass

from prowl import ProwlResult
from src.agent.ollama import LLMResult, OllamaClient
from support import SupportCard

@dataclass(frozen=True)
class RenderedAnswer:
    text: str
    route: str | None = None


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
    parts.append(
        "Reply with only one concise Discord-ready final answer. Do not ask follow-up or clarifying questions. "
        "Do not add facts beyond this evidence. For a contributor question, name whether the command is for a "
        "packaged install or a dev checkout. For state-changing recovery, lead with the safest read-only check "
        "unless the user explicitly asks to perform that action."
    )
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
        route = "lfm" if sources is not None and sources.status == "ok" else "gemma"
        result: LLMResult = await self.client.answer(_prompt(query, card, sources), route=route)
        if result.status != "ok" or "?" in result.text:
            return RenderedAnswer(card.answer)
        return RenderedAnswer(result.text, route=route)
