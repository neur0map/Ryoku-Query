from __future__ import annotations

import re
from pathlib import Path

import discord

from prowl import ProwlClient, is_source_query
from src.agent.answering import Answerer
from src.agent.replies import (
    clarification_embed,
    no_match_embed,
    safety_embed,
    source_embed,
    support_embed,
    support_view,
    unavailable_embed,
)
from support import SupportRetriever, requires_safety_confirmation


def extract_query(
    message,
    bot_user_id: int,
    support_channel_id: int | None = None,
) -> str | None:
    if message.author.bot:
        return None
    in_support_channel = (
        support_channel_id is not None
        and support_channel_id > 0
        and getattr(getattr(message, "channel", None), "id", None) == support_channel_id
    )
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
    if not in_support_channel and not mentioned and not replied:
        return None
    content = re.sub(fr"<@!?{bot_user_id}>", "", message.content).strip()
    return content or None


def should_search(query: str) -> bool:
    return is_source_query(query)


async def handle_message(
    message,
    bot_user_id: int,
    retriever: SupportRetriever,
    prowl: ProwlClient | None,
    logo_path: Path,
    *,
    answerer: Answerer | None = None,
    support_channel_id: int | None = None,
) -> bool:
    query = extract_query(message, bot_user_id, support_channel_id)
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
        )
    ):
        source_hints = (
            decision.card.source_hints
            if decision.kind == "answer" and decision.card
            else ()
        )
        source_result = await prowl.search(query, source_hints)

    rendered_answer = None
    if (
        not dangerous
        and answerer is not None
        and decision.kind == "answer"
        and decision.card is not None
    ):
        rendered_answer = await answerer.render(query, decision.card, source_result)

    view = None
    if dangerous:
        embed = safety_embed()
    elif source_result is not None and source_result.status == "ok":
        embed = source_embed(
            source_result,
            rendered_answer.text if rendered_answer is not None else None,
        )
    elif source_result is not None and source_result.status == "unavailable":
        embed = unavailable_embed(source_result.error)
    elif source_result is not None:
        embed = no_match_embed()
    elif decision.kind == "answer" and decision.card is not None:
        embed = support_embed(
            decision.card,
            rendered_answer.text if rendered_answer is not None else None,
        )
        view = support_view(decision.card)
    elif decision.kind == "clarify":
        if len(decision.alternatives) == 1:
            card = decision.alternatives[0]
            embed = support_embed(card)
            view = support_view(card)
        else:
            embed = clarification_embed(decision)
    else:
        embed = no_match_embed()

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
