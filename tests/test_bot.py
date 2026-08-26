import os
import unittest
from pathlib import Path
from unittest.mock import patch

from bot import extract_query, handle_message, load_config
from prowl import ProwlResult, SourceHit
from support import RankedIntent, RetrievalDecision, SupportCard


class User:
    def __init__(self, user_id, bot=False):
        self.id = user_id
        self.bot = bot

    def __str__(self):
        return f"user-{self.id}"


class Reference:
    def __init__(self, author):
        self.resolved = type("Resolved", (), {"author": author})()


class Channel:
    def __init__(self):
        self.calls = []

    async def send(self, **kwargs):
        self.calls.append(kwargs)


class Message:
    def __init__(
        self,
        content,
        *,
        author=None,
        mentions=(),
        reference=None,
    ):
        self.content = content
        self.author = author or User(7)
        self.mentions = list(mentions)
        self.reference = reference
        self.channel = Channel()


class Retriever:
    def __init__(self, decision):
        self.decision = decision
        self.queries = []

    def retrieve(self, query):
        self.queries.append(query)
        return self.decision


class Prowl:
    def __init__(self, result):
        self.result = result
        self.queries = []
        self.hints = []

    async def search(self, query, source_hints=()):
        self.queries.append(query)
        self.hints.append(source_hints)
        return self.result


def support_card(**overrides):
    values = {
        "id": "health.doctor",
        "title": "Run Ryoku Doctor",
        "examples": ("check ryoku", "repair drift"),
        "keywords": ("doctor",),
        "exact_terms": ("ryoku doctor",),
        "answer": "Run `ryoku doctor` to inspect supported drift.",
        "risk": "informational",
        "docs_url": "https://docs.ryoku.dev/docs/troubleshoot",
        "source_hints": ("docs/cli.md",),
        "clarifies_with": (),
    }
    values.update(overrides)
    return SupportCard(**values)


class InvocationTests(unittest.TestCase):
    def test_ordinary_message_is_ignored(self):
        self.assertIsNone(extract_query(Message("hello"), bot_user_id=42))

    def test_bot_message_is_ignored(self):
        message = Message("<@42> hello", author=User(42, bot=True))
        self.assertIsNone(extract_query(message, bot_user_id=42))

    def test_mention_is_removed_from_query(self):
        message = Message(
            "<@42>   how do I run doctor?",
            mentions=(User(42, bot=True),),
        )
        self.assertEqual(
            extract_query(message, bot_user_id=42),
            "how do I run doctor?",
        )

    def test_reply_to_bot_is_handled(self):
        message = Message(
            "where is that implemented?",
            reference=Reference(User(42, bot=True)),
        )
        self.assertEqual(
            extract_query(message, bot_user_id=42),
            "where is that implemented?",
        )

    def test_reply_to_other_user_is_ignored(self):
        message = Message("hello", reference=Reference(User(9)))
        self.assertIsNone(extract_query(message, bot_user_id=42))



class ConfigTests(unittest.TestCase):
    def test_loads_stable_repository_configuration(self):
        values = {
            "TOKEN": "secret",
            "RYOKU_REPO_PATH": "/srv/ryoku-stable",
            "SUPPORT_PATH": "data/support.json",
            "PROWL_TIMEOUT_SECONDS": "3.5",
            "PROWL_RESULT_LIMIT": "12",
        }
        with patch("bot.load_dotenv"), patch.dict(
            os.environ, values, clear=True
        ):
            config = load_config()

        self.assertEqual(config.token, "secret")
        self.assertEqual(config.ryoku_repo_path, Path("/srv/ryoku-stable"))
        self.assertEqual(config.prowl_timeout, 3.5)
        self.assertEqual(config.prowl_result_limit, 12)

    def test_requires_token(self):
        with patch("bot.load_dotenv"), patch.dict(
            os.environ, {}, clear=True
        ), self.assertRaisesRegex(RuntimeError, "Missing TOKEN"):
            load_config()

    def test_rejects_unbounded_prowl_configuration(self):
        values = {
            "TOKEN": "secret",
            "PROWL_TIMEOUT_SECONDS": "31",
            "PROWL_RESULT_LIMIT": "21",
        }
        with patch("bot.load_dotenv"), patch.dict(
            os.environ, values, clear=True
        ), self.assertRaisesRegex(
            RuntimeError, "PROWL_TIMEOUT_SECONDS"
        ):
            load_config()

class HandlerTests(unittest.IsolatedAsyncioTestCase):
    async def test_curated_answer_is_built_then_sent_once(self):
        card = support_card()
        retriever = Retriever(RetrievalDecision("answer", card=card))
        message = Message(
            "<@42> doctor please", mentions=(User(42, bot=True),)
        )

        handled = await handle_message(
            message,
            bot_user_id=42,
            retriever=retriever,
            prowl=None,
            logo_path=Path("data/logo.png"),
        )

        self.assertTrue(handled)
        self.assertEqual(len(message.channel.calls), 1)
        call = message.channel.calls[0]
        self.assertEqual(call["embed"].title, card.title)
        self.assertEqual(call["embed"].footer.text, "Ryoku support")
        self.assertEqual(call["embed"].fields[0].name, "Documentation")
        self.assertIn("file", call)
        self.assertFalse(call["allowed_mentions"].everyone)
        self.assertFalse(call["allowed_mentions"].users)
        self.assertFalse(call["allowed_mentions"].roles)
        call["file"].close()

    async def test_ambiguity_sends_one_question(self):
        first = support_card(id="health.report", title="Create report")
        second = support_card(id="health.privacy", title="Report privacy")
        retriever = Retriever(
            RetrievalDecision("clarify", alternatives=(first, second))
        )
        message = Message("question", reference=Reference(User(42, bot=True)))

        await handle_message(message, 42, retriever, None, Path("missing.png"))

        self.assertEqual(len(message.channel.calls), 1)
        description = message.channel.calls[0]["embed"].description
        self.assertIn("Create report", description)
        self.assertIn("Report privacy", description)

    async def test_source_query_uses_prowl_once_and_cites_result(self):
        primary = support_card(
            id="overview.repository",
            source_hints=("docs/structure.md",),
        )
        secondary = support_card(
            id="settings.backend",
            source_hints=("ryoku/cli/main.go",),
        )
        retriever = Retriever(
            RetrievalDecision(
                "answer",
                card=primary,
                ranked=(
                    RankedIntent(
                        primary, 0.2, 0, 4, 0.5, ("lexical", "semantic")
                    ),
                    RankedIntent(
                        secondary, 0.19, 0, 3, 0.4, ("lexical", "semantic")
                    ),
                ),
            )
        )
        prowl = Prowl(
            ProwlResult(
                "ok",
                hits=(
                    SourceHit(
                        "ryoku/cli/main.go",
                        1,
                        32,
                        "package main",
                        False,
                    ),
                ),
            )
        )
        message = Message(
            "<@42> where is the CLI implemented?",
            mentions=(User(42, bot=True),),
        )

        await handle_message(message, 42, retriever, prowl, Path("missing.png"))

        self.assertEqual(prowl.queries, ["where is the CLI implemented?"])
        self.assertEqual(prowl.hints, [("docs/structure.md",)])
        self.assertEqual(len(message.channel.calls), 1)
        embed = message.channel.calls[0]["embed"]
        self.assertIn("ryoku/cli/main.go:1-32", embed.fields[0].value)

    async def test_dangerous_source_request_never_calls_prowl(self):
        retriever = Retriever(RetrievalDecision("no_match"))
        prowl = Prowl(ProwlResult("ok"))
        message = Message(
            "<@42> find source and wipe everything without checking",
            mentions=(User(42, bot=True),),
        )

        await handle_message(message, 42, retriever, prowl, Path("missing.png"))

        self.assertEqual(prowl.queries, [])
        self.assertEqual(len(message.channel.calls), 1)
        self.assertIn(
            "won't recommend a destructive action",
            message.channel.calls[0]["embed"].description,
        )

    async def test_unrelated_mention_does_not_call_prowl(self):
        card = support_card(id="hardware.gpu-detect")
        retriever = Retriever(
            RetrievalDecision(
                "no_match",
                ranked=(
                    RankedIntent(
                        card=card,
                        fused_score=0.136,
                        exact_score=0,
                        lexical_score=5.7,
                        semantic_score=0.19,
                        channels=("lexical",),
                    ),
                ),
            )
        )
        prowl = Prowl(ProwlResult("ok"))
        message = Message(
            "<@42> can you recommend dinner?",
            mentions=(User(42, bot=True),),
        )

        await handle_message(message, 42, retriever, prowl, Path("missing.png"))

        self.assertEqual(prowl.queries, [])
        self.assertEqual(len(message.channel.calls), 1)
        self.assertIn("couldn't find", message.channel.calls[0]["embed"].description)


if __name__ == "__main__":
    unittest.main()
