from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass
from pathlib import Path

import discord
from dotenv import load_dotenv
from model2vec import StaticModel

from prowl import ProwlClient, ProwlResult, is_source_query
from support import (
    RetrievalDecision,
    SupportCard,
    SupportRetriever,
    load_support_cards,
    normalize_text,
    requires_safety_confirmation,
)


@dataclass(frozen=True)
class Config:
    token: str
    model_name: str
    support_path: Path
    ryoku_repo_path: Path | None
    prowl_timeout: float
    prowl_result_limit: int


def load_config() -> Config:
    load_dotenv()
    token = os.getenv("TOKEN", "").strip()
    if not token:
        raise RuntimeError("Missing TOKEN in .env")
    timeout = float(os.getenv("PROWL_TIMEOUT_SECONDS", "4"))
    limit = int(os.getenv("PROWL_RESULT_LIMIT", "20"))
    if not 0.1 <= timeout <= 30:
        raise RuntimeError("PROWL_TIMEOUT_SECONDS must be between 0.1 and 30")
    if not 1 <= limit <= 20:
        raise RuntimeError("PROWL_RESULT_LIMIT must be between 1 and 20")
    repo = os.getenv("RYOKU_REPO_PATH", "").strip()
    return Config(
        token=token,
        model_name=os.getenv(
            "MODEL_NAME", "minishlab/potion-base-32M"
        ),
        support_path=Path(os.getenv("SUPPORT_PATH", "data/support.json")),
        ryoku_repo_path=Path(repo) if repo else None,
        prowl_timeout=timeout,
        prowl_result_limit=limit,
    )


def extract_query(message, bot_user_id: int) -> str | None:
    if message.author.bot:
        return None
    mentioned = any(
        getattr(user, "id", None) == bot_user_id
        for user in getattr(message, "mentions", ())
    )
    resolved = getattr(getattr(message, "reference", None), "resolved", None)
    replied = (
        resolved is not None
        and getattr(getattr(resolved, "author", None), "id", None)
        == bot_user_id
    )
    if not mentioned and not replied:
        return None
    content = re.sub(fr"<@!?{bot_user_id}>", "", message.content).strip()
    return content or None


def _base_embed(title: str, description: str) -> discord.Embed:
    return discord.Embed(
        title=title,
        description=description,
        color=0xEA5322,
    )


def _support_embed(card: SupportCard) -> discord.Embed:
    embed = _base_embed(card.title, card.answer)
    embed.set_footer(text="Ryoku support")
    return embed


def _support_view(card: SupportCard) -> discord.ui.View:
    view = discord.ui.View(timeout=None)
    safety = {
        "informational": "🟢 Read-only / informational",
        "state-changing": "🟡 Changes system state",
        "destructive": "🔴 Potentially destructive",
    }.get(card.risk)
    if safety is not None:
        view.add_item(
            discord.ui.Button(
                label=safety,
                style=discord.ButtonStyle.secondary,
                disabled=True,
            )
        )
    if card.docs_url:
        view.add_item(
            discord.ui.Button(
                label="Open Ryoku documentation →",
                style=discord.ButtonStyle.link,
                url=card.docs_url,
            )
        )
    return view


def _clarification_embed(decision: RetrievalDecision) -> discord.Embed:
    choices = "\n".join(
        f"- **{card.title}**" for card in decision.alternatives
    )
    description = f"Which of these do you mean?\n{choices}"
    return _base_embed("One detail first", description)


def _safety_embed() -> discord.Embed:
    return _base_embed(
        "I need more context",
        "I won't recommend a destructive action from a broad source match. "
        "Describe the symptom, whether the desktop or a TTY still works, and "
        "what recovery steps you already tried.",
    )


def _source_embed(result: ProwlResult) -> discord.Embed:
    embed = _base_embed(
        "Ryoku stable source",
        "I found these locations in the indexed stable Ryoku repository.",
    )
    for hit in result.hits:
        label = "Potentially destructive source" if hit.risky else "Source"
        snippet = hit.snippet or "Matched source location"
        embed.add_field(
            name=label,
            value=f"**`{hit.citation}`**\n{snippet}",
            inline=False,
        )
    embed.set_footer(text="Ryoku source")
    return embed


def _unavailable_embed(error: str | None) -> discord.Embed:
    detail = error or "The stable source index is unavailable."
    return _base_embed(
        "Source search unavailable",
        f"I couldn't search the stable Ryoku source right now: {detail}",
    )


def _no_match_embed() -> discord.Embed:
    return _base_embed(
        "No confident Ryoku answer",
        "I couldn't find a reviewed answer for that question. Try naming the "
        "Ryoku component, command, or error you are seeing.",
    )


def _should_search(query: str) -> bool:
    return "ryoku" in normalize_text(query)


async def handle_message(
    message,
    bot_user_id: int,
    retriever: SupportRetriever,
    prowl: ProwlClient | None,
    logo_path: Path,
) -> bool:
    query = extract_query(message, bot_user_id)
    if query is None:
        return False
    decision = retriever.retrieve(query)
    dangerous = requires_safety_confirmation(query)
    source_result = None
    if (
        not dangerous
        and prowl is not None
        and (
            is_source_query(query)
            or (
                decision.kind == "no_match"
                and _should_search(query)
            )
        )
    ):
        source_hints = (
            decision.card.source_hints
            if decision.kind == "answer" and decision.card
            else ()
        )
        source_result = await prowl.search(query, source_hints)

    view = None
    if dangerous:
        embed = _safety_embed()
    elif source_result is not None and source_result.status == "ok":
        embed = _source_embed(source_result)
    elif source_result is not None and source_result.status == "unavailable":
        embed = _unavailable_embed(source_result.error)
    elif source_result is not None:
        embed = _no_match_embed()
    elif decision.kind == "answer" and decision.card is not None:
        embed = _support_embed(decision.card)
        view = _support_view(decision.card)
    elif decision.kind == "clarify":
        if len(decision.alternatives) == 1:
            card = decision.alternatives[0]
            embed = _support_embed(card)
            view = _support_view(card)
        else:
            embed = _clarification_embed(decision)
    else:
        embed = _no_match_embed()

    send = {
        "embed": embed,
        "allowed_mentions": discord.AllowedMentions.none(),
    }
    if view is not None:
        send["view"] = view
    if logo_path.is_file():
        send["file"] = discord.File(str(logo_path), filename="logo.png")
        embed.set_author(
            name="Ryoku Help",
            icon_url="attachment://logo.png",
        )
    await message.reply(
        **send,
        mention_author=True,
    )
    return True


async def run(config: Config) -> None:
    model = StaticModel.from_pretrained(config.model_name)
    cards = load_support_cards(config.support_path)
    retriever = SupportRetriever(cards, model)
    prowl = (
        ProwlClient(
            config.ryoku_repo_path,
            timeout=config.prowl_timeout,
            result_limit=config.prowl_result_limit,
        )
        if config.ryoku_repo_path is not None
        else None
    )
    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)

    @client.event
    async def on_ready():
        print(
            f"Logged in as {client.user} (ID: {client.user.id}); "
            f"{len(cards)} support intents; "
            f"Prowl {'enabled' if prowl else 'disabled'}"
        )

    @client.event
    async def on_message(message):
        await handle_message(
            message,
            client.user.id,
            retriever,
            prowl,
            Path("data/logo.png"),
        )

    async with client:
        await client.start(config.token)


def main() -> None:
    try:
        asyncio.run(run(load_config()))
    except KeyboardInterrupt:
        print("\nRyoku Help stopped.")


if __name__ == "__main__":
    main()
