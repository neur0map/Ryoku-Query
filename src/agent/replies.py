from __future__ import annotations

import discord

from feedback import FeedbackStore, FeedbackView
from prowl import ProwlResult
from support import SupportCard

BRAND_COLOR = 0xEA5322
BOT_NAME = "Nero"

def base_embed(title: str, description: str) -> discord.Embed:
    return discord.Embed(
        title=title,
        description=description,
        color=BRAND_COLOR,
    )


def support_embed(card: SupportCard, answer: str | None = None) -> discord.Embed:
    embed = base_embed(card.title, answer or card.answer)
    embed.set_footer(text=f"{BOT_NAME} • Ryoku support")
    return embed


def support_view(
    card: SupportCard, feedback: FeedbackStore | None = None
) -> discord.ui.View:
    view = FeedbackView(feedback) if feedback is not None else discord.ui.View(timeout=None)
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


def safety_embed() -> discord.Embed:
    return base_embed(
        "I need more context",
        "I won't recommend a destructive action from a broad source match. "
        "Describe the symptom, whether the desktop or a TTY still works, and "
        "what recovery steps you already tried.",
    )


def source_embed(result: ProwlResult, answer: str | None = None) -> discord.Embed:
    embed = base_embed(
        "Ryoku stable source",
        answer or "I found these locations in the indexed stable Ryoku repository.",
    )
    for hit in result.hits:
        label = "Potentially destructive source" if hit.risky else "Source"
        snippet = hit.snippet or "Matched source location"
        embed.add_field(
            name=label,
            value=f"**`{hit.citation}`**\n{snippet}",
            inline=False,
        )
    embed.set_footer(text=f"{BOT_NAME} • Ryoku source")
    return embed


def unavailable_embed(error: str | None) -> discord.Embed:
    detail = error or "The stable source index is unavailable."
    return base_embed(
        "Source search unavailable",
        f"I couldn't search the stable Ryoku source right now: {detail}",
    )


def no_match_embed() -> discord.Embed:
    return base_embed(
        "No confident Ryoku answer",
        "I couldn't find a reviewed answer for that question. Try naming the "
        "Ryoku component, command, or error you are seeing.",
    )
