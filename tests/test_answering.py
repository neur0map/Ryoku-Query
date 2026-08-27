import unittest

from prowl import ProwlResult, SourceHit
from src.agent.answering import Answerer
from src.agent.ollama import LLMResult
from support import SupportCard


class LLM:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def answer(self, prompt, *, route):
        self.calls.append((prompt, route))
        return self.result


def card(risk="informational"):
    return SupportCard(
        id="health.status",
        title="Check status",
        examples=("check status", "is it healthy"),
        keywords=("status",),
        exact_terms=("ryoku status",),
        answer="Run `ryoku status` for a read-only health summary.",
        risk=risk,
        docs_url="https://docs.ryoku.dev/docs/troubleshoot",
        source_hints=("docs/cli.md",),
        clarifies_with=(),
    )


class AnswererTests(unittest.IsolatedAsyncioTestCase):
    async def test_curated_answer_uses_gemma_with_reviewed_evidence(self):
        llm = LLM(LLMResult("ok", text="Use `ryoku status` to check it."))
        answerer = Answerer(llm)

        result = await answerer.render("how do I check health?", card())

        self.assertEqual(result.text, "Use `ryoku status` to check it.")
        self.assertEqual(result.route, "gemma")
        prompt, route = llm.calls[0]
        self.assertEqual(route, "gemma")
        self.assertIn("Run `ryoku status`", prompt)

    async def test_cited_diagnostic_uses_lfm_and_never_includes_thinking(self):
        llm = LLM(LLMResult("ok", text="Check the cited service log first."))
        answerer = Answerer(llm)
        sources = ProwlResult("ok", hits=(SourceHit("docs/cli.md", 1, 20, "Ryoku status", False),))

        result = await answerer.render("my shell failed after update", card(), sources)

        self.assertEqual(result.route, "lfm")
        self.assertNotIn("thinking", result.text.lower())
        self.assertEqual(llm.calls[0][1], "lfm")
        self.assertIn("docs/cli.md:1-20", llm.calls[0][0])

    async def test_cited_contributor_question_uses_lfm_without_error_words(self):
        llm = LLM(LLMResult("ok", text="Deploy from the checkout after testing."))
        answerer = Answerer(llm)
        sources = ProwlResult(
            "ok", hits=(SourceHit("docs/development.md", 7, 16, "deploy then test", False),)
        )

        result = await answerer.render(
            "Where is the supported contributor flow for deploying custom work?",
            card(),
            sources,
        )

        self.assertEqual(result.route, "lfm")
        self.assertEqual(llm.calls[0][1], "lfm")

    async def test_destructive_card_does_not_call_model(self):
        llm = LLM(LLMResult("ok", text="unsafe"))
        answerer = Answerer(llm)

        result = await answerer.render("wipe it", card("destructive"))

        self.assertEqual(result.text, "Run `ryoku status` for a read-only health summary.")
        self.assertEqual(llm.calls, [])

    async def test_unavailable_model_falls_back_to_reviewed_answer(self):
        llm = LLM(LLMResult("unavailable", error="offline"))
        answerer = Answerer(llm)

        result = await answerer.render("health?", card())

        self.assertEqual(result.text, "Run `ryoku status` for a read-only health summary.")
        self.assertEqual(result.route, None)
