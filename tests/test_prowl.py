import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from prowl import ProwlClient, is_source_query


class RecordingRunner:
    def __init__(self, stdout=b"[]", returncode=0, error=None):
        self.stdout = stdout
        self.returncode = returncode
        self.error = error
        self.calls = []

    async def __call__(self, argv, cwd, timeout, max_output):
        self.calls.append((argv, cwd, timeout, max_output))
        if self.error is not None:
            raise self.error
        return self.returncode, self.stdout, b""



class DispatchRunner:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    async def __call__(self, argv, cwd, timeout, max_output):
        self.calls.append((argv, cwd, timeout, max_output))
        return 0, json.dumps(self.responses[argv[1]]).encode(), b""


class ProwlClientTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.repo = Path(self.directory.name)

    def search(self, client, query, **kwargs):
        return asyncio.run(client.search(query, **kwargs))

    def test_passes_user_query_as_one_argument_without_smart(self):
        runner = RecordingRunner()
        client = ProwlClient(self.repo, runner=runner)
        query = 'source; rm -rf "$HOME"'

        self.search(client, query)

        argv, cwd, _, _ = runner.calls[0]
        executable = argv[0].lower()
        self.assertTrue(
            executable.endswith("prowl-agent")
            or executable.endswith("prowl-agent.exe")
        )
        self.assertEqual(argv[1:3], ("search", query))
        self.assertNotIn("--smart", argv)
        self.assertEqual(cwd, self.repo)

    def test_appends_smart_flag_when_enabled(self):
        runner = RecordingRunner()
        client = ProwlClient(self.repo, smart_search=True, runner=runner)

        self.search(client, "where is the CLI implemented?")

        argv = runner.calls[0][0]
        executable = argv[0].lower()
        self.assertTrue(
            executable.endswith("prowl-agent")
            or executable.endswith("prowl-agent.exe")
        )
        self.assertEqual(argv[1:3], (
            "search",
            "where is the CLI implemented?",
        ))
        self.assertIn("--smart", argv)

    def test_bounds_result_limit(self):
        runner = RecordingRunner()
        client = ProwlClient(self.repo, result_limit=200, runner=runner)

        self.search(client, "query")

        argv = runner.calls[0][0]
        self.assertEqual(argv[argv.index("--limit") + 1], "20")

    def test_prefers_docs_to_workflows_for_support_query(self):
        payload = [
            {
                "file": ".github/workflows/recovery.yml",
                "start_line": 1,
                "end_line": 10,
                "snippet": "recovery workflow",
            },
            {
                "file": "docs/updates.md",
                "start_line": 40,
                "end_line": 60,
                "snippet": "stable update guidance",
            },
        ]
        runner = RecordingRunner(json.dumps(payload).encode())
        client = ProwlClient(self.repo, runner=runner)

        result = self.search(client, "how do stable updates work")

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.hits[0].file, "docs/updates.md")

    def test_accepts_search_json_matches_payload(self):
        payload = {
            "query": "where",
            "matches": [
                {
                    "file": "ryoku/cli/main.go",
                    "start_line": 1,
                    "end_line": 32,
                    "snippet": "package main",
                }
            ],
        }
        runner = RecordingRunner(json.dumps(payload).encode())
        client = ProwlClient(self.repo, runner=runner)

        result = self.search(client, "where is the CLI implemented?")

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.hits[0].file, "ryoku/cli/main.go")

    def test_direct_path_match_is_not_demoted(self):
        payload = [
            {
                "file": "docs/cli.md",
                "start_line": 1,
                "end_line": 20,
                "snippet": "general CLI guide",
            },
            {
                "file": "ryoku/cli/main.go",
                "start_line": 1,
                "end_line": 32,
                "snippet": "package main",
            },
        ]
        runner = RecordingRunner(json.dumps(payload).encode())
        client = ProwlClient(self.repo, runner=runner)

        result = self.search(client, "show ryoku/cli/main.go")

        self.assertEqual(result.hits[0].file, "ryoku/cli/main.go")

    def test_rejects_result_when_every_source_hit_is_risky(self):
        payload = [
            {
                "file": "bin/ryoku-recovery",
                "start_line": 140,
                "end_line": 160,
                "snippet": 'rm -rf "$RCFG/ryoku/user_edits"',
            }
        ]
        runner = RecordingRunner(json.dumps(payload).encode())
        client = ProwlClient(self.repo, runner=runner)

        result = self.search(client, "where is recovery implemented")

        self.assertEqual(result.status, "no_match")
        self.assertEqual(result.hits, ())

    def test_malformed_json_returns_unavailable(self):
        client = ProwlClient(
            self.repo, runner=RecordingRunner(stdout=b"not-json")
        )

        result = self.search(client, "query")

        self.assertEqual(result.status, "unavailable")
        self.assertIn("JSON", result.error)

    def test_timeout_returns_unavailable(self):
        runner = RecordingRunner(error=TimeoutError())
        client = ProwlClient(self.repo, runner=runner)

        result = self.search(client, "query")

        self.assertEqual(result.status, "unavailable")
        self.assertEqual(result.error, "Prowl search timed out")

    def test_reads_explicit_repository_path_without_unsupported_peek_command(self):
        target = self.repo / "ryoku/hyprland/hyprland.lua"
        target.parent.mkdir(parents=True)
        target.write_text("return require('modules')\n", encoding="utf-8")
        runner = RecordingRunner()
        client = ProwlClient(self.repo, runner=runner)

        result = self.search(
            client,
            "Show ryoku/hyprland/hyprland.lua from the stable source.",
        )

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.hits[0].file, "ryoku/hyprland/hyprland.lua")
        self.assertEqual(result.hits[0].snippet, "return require('modules')")
        self.assertEqual(runner.calls, [])

    def test_reviewed_hint_only_reranks_actual_search_results(self):
        payload = [
            {
                "file": "ryoku/hub/backend/other.go",
                "start_line": 1,
                "end_line": 20,
                "snippet": "other backend code",
            },
            {
                "file": "ryoku/hub/backend/main.go",
                "start_line": 1,
                "end_line": 20,
                "snippet": "package main",
            },
        ]
        runner = RecordingRunner(json.dumps(payload).encode())
        client = ProwlClient(self.repo, runner=runner)

        result = self.search(
            client,
            "Where is the Settings backend implementation?",
            source_hints=("ryoku/hub/backend/main.go",),
        )

        self.assertEqual(result.hits[0].file, "ryoku/hub/backend/main.go")
        self.assertEqual([call[0][1] for call in runner.calls], ["search"])

    def test_uses_prowl_find_for_explicit_symbol(self):
        runner = DispatchRunner(
            {
                "search": [],
                "find": [
                    {
                        "name": "stepVerify",
                        "signature": "func stepVerify(e *engine) error",
                        "file": "ryoku-shell-installer/engine.go",
                        "line": 1246,
                        "end_line": 1319,
                    }
                ],
            }
        )
        client = ProwlClient(self.repo, runner=runner)

        result = self.search(
            client,
            "Where is `stepVerify` defined?",
        )

        self.assertEqual(
            result.hits[0].file, "ryoku-shell-installer/engine.go"
        )
        self.assertIn("find", [call[0][1] for call in runner.calls])

    def test_uses_narrow_candidate_path_for_update_document_query(self):
        target = self.repo / "docs/updates.md"
        target.parent.mkdir(parents=True)
        target.write_text("stable update guidance\n", encoding="utf-8")
        runner = RecordingRunner()
        client = ProwlClient(self.repo, runner=runner)

        result = self.search(
            client,
            "Find the stable source document that defines update delivery.",
        )

        self.assertEqual(result.hits[0].file, "docs/updates.md")
        self.assertEqual(result.hits[0].snippet, "stable update guidance")
        self.assertEqual(runner.calls, [])


    def test_source_query_detection_is_narrow(self):
        self.assertTrue(is_source_query("Where is the CLI implemented?"))
        self.assertTrue(is_source_query("show ryoku/cli/main.go"))
        self.assertFalse(is_source_query("my bar disappeared after login"))
        self.assertFalse(
            is_source_query(
                "How do I publish a package to the Arch User Repository?"
            )
        )
        self.assertFalse(
            is_source_query("Where is the diagnostic report file saved?")
        )
        self.assertFalse(
            is_source_query(
                "I edited the source configuration, but the desktop is stale."
            )
        )


if __name__ == "__main__":
    unittest.main()
