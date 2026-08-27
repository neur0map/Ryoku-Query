import asyncio
import unittest

from src.agent.ollama import OllamaClient


class Runner:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def __call__(self, url, payload, timeout):
        self.calls.append((url, payload, timeout))
        return self.response


class OllamaClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_gemma_disables_thinking_and_limits_residency(self):
        runner = Runner({"message": {"content": "  Run `ryoku status`.  "}})
        client = OllamaClient("http://127.0.0.1:11434", "gemma4:e4b", "lfm2.5:latest", runner=runner)

        result = await client.answer("facts", route="gemma")

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.text, "Run `ryoku status`.")
        _, payload, _ = runner.calls[0]
        self.assertEqual(payload["model"], "gemma4:e4b")
        self.assertFalse(payload["think"])
        self.assertEqual(payload["keep_alive"], "0")

    async def test_lfm_discards_private_thinking(self):
        runner = Runner({"message": {"content": "Run `ryoku doctor --check`.", "thinking": "private chain"}})
        client = OllamaClient("http://127.0.0.1:11434", "gemma4:e4b", "lfm2.5:latest", runner=runner)

        result = await client.answer("facts", route="lfm")

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.text, "Run `ryoku doctor --check`.")
        _, payload, _ = runner.calls[0]
        self.assertEqual(payload["model"], "lfm2.5:latest")
        self.assertTrue(payload["think"])
        self.assertNotIn("private chain", result.text)

    async def test_serializes_requests_to_keep_two_models_out_of_memory(self):
        active = 0
        peak = 0

        async def slow_runner(url, payload, timeout):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1
            return {"message": {"content": "ok"}}

        client = OllamaClient("http://ollama", "gemma", "lfm", runner=slow_runner)
        await asyncio.gather(
            client.answer("one", route="gemma"),
            client.answer("two", route="lfm"),
        )
        self.assertEqual(peak, 1)

    async def test_strips_embedded_thinking_markup(self):
        runner = Runner({"message": {"content": "<think>private</think>\nUse `ryoku status`."}})
        client = OllamaClient("http://127.0.0.1:11434", "gemma4:e4b", "lfm2.5:latest", runner=runner)

        result = await client.answer("facts", route="gemma")

        self.assertEqual(result.text, "Use `ryoku status`.")

    async def test_invalid_response_is_unavailable(self):
        runner = Runner({"message": {"thinking": "no final answer"}})
        client = OllamaClient("http://127.0.0.1:11434", "gemma4:e4b", "lfm2.5:latest", runner=runner)

        result = await client.answer("facts", route="gemma")

        self.assertEqual(result.status, "unavailable")
        self.assertIn("final answer", result.error)
