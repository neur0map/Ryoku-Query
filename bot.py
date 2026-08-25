import asyncio
import json
import os
from pathlib import Path

import discord
import numpy as np
from dotenv import load_dotenv
from model2vec import StaticModel


load_dotenv()

TOKEN = os.getenv("TOKEN")
THRESHOLD = float(os.getenv("THRESHOLD", "0.70"))
MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "minishlab/potion-code-16M-v2",
)
FAQ_PATH = Path(os.getenv("FAQ_PATH", "faq.json"))


if not TOKEN:
    raise RuntimeError("Missing TOKEN in .env")

if not 0.0 <= THRESHOLD <= 1.0:
    raise RuntimeError(
        "THRESHOLD must be between 0.0 and 1.0"
    )


def load_faq(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        entries = json.load(f)

    if not isinstance(entries, list) or not entries:
        raise RuntimeError(
            "faq.json must contain a non-empty JSON list."
        )

    for i, entry in enumerate(entries):
        if "text" not in entry or "response" not in entry:
            raise RuntimeError(
                f"FAQ entry #{i} must contain "
                "'text' and 'response'."
            )

    return entries


def normalize_rows(
    vectors: np.ndarray,
) -> np.ndarray:
    vectors = np.asarray(
        vectors,
        dtype=np.float32,
    )

    norms = np.linalg.norm(
        vectors,
        axis=1,
        keepdims=True,
    )

    norms = np.clip(
        norms,
        1e-12,
        None,
    )

    return vectors / norms


print(f"Loading embedding model: {MODEL_NAME}")

model = StaticModel.from_pretrained(
    MODEL_NAME
)

faq = load_faq(FAQ_PATH)

faq_texts = [
    entry["text"]
    for entry in faq
]

print(
    f"Embedding {len(faq_texts)} "
    "FAQ/example entries..."
)

faq_vectors = normalize_rows(
    model.encode(faq_texts)
)


intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(
    intents=intents
)


def find_best_match(
    message_text: str,
) -> tuple[dict, float]:

    query_vector = normalize_rows(
        model.encode([message_text])
    )[0]

    # Vectors are normalized,
    # so dot product == cosine similarity.
    scores = faq_vectors @ query_vector

    best_index = int(
        np.argmax(scores)
    )

    best_score = float(
        scores[best_index]
    )

    return (
        faq[best_index],
        best_score,
    )


@client.event
async def on_ready():
    print(
        f"Logged in as {client.user} "
        f"(ID: {client.user.id})"
    )

    print(
        f"Threshold: {THRESHOLD:.2f}"
    )

    print(
        "Ryoku Help is listening."
    )


@client.event
async def on_message(
    message: discord.Message,
):
    if message.author.bot:
        return

    content = message.content.strip()

    if not content:
        return

    match, score = find_best_match(
        content
    )

    print(
        f"[match={score:.3f}] "
        f"{message.author}: "
        f"{content!r} "
        f"-> {match['text']!r}"
    )

    if score < THRESHOLD:
        return

    embed = discord.Embed(
        title=match.get(
            "title",
            "Ryoku Help",
        ),
        description=match["response"],
        url=match.get("docs_url"),
        color=0xEA5322,
    )

    file = discord.File(
        "./data/logo.png",
        filename="logo.png",
    )

    embed.set_author(
        name="Ryoku Help",
        icon_url="attachment://logo.png",
    )

    await message.channel.send(
        file=file,
        embed=embed,
        allowed_mentions=discord.AllowedMentions.none(),
    )

    if match.get("docs_url"):
        embed.add_field(
            name="Documentation",
            icon_url="./data/logo.png",
            value=(
                "[Open Ryoku Docs]"
                f"({match['docs_url']})"
            ),
            inline=False,
        )

    embed.set_footer(
        text=(
            f"Similarity: {score:.1%}"
        )
    )

    await message.channel.send(
        embed=embed,
        allowed_mentions=(
            discord.AllowedMentions.none()
        ),
    )


async def main():
    try:
        async with client:
            await client.start(TOKEN)

    finally:
        if not client.is_closed():
            await client.close()

        print(
            "Discord client closed."
        )


if __name__ == "__main__":
    try:
        asyncio.run(main())

    except KeyboardInterrupt:
        print(
            "\nRyoku Help stopped."
        )
