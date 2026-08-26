import json
import tempfile
import unittest
from pathlib import Path

from benchmark import (
    BenchmarkCase,
    CaseResult,
    HybridEvaluator,
    case_passes,
    load_benchmark,
    score_results,
)
from prowl import ProwlResult, SourceHit
from support import RetrievalDecision, SupportCard


class BenchmarkScoreTests(unittest.TestCase):
    def test_case_pass_requires_every_constraint(self):
        case = BenchmarkCase(
            id="easy-doctor",
            query="run a health check",
            difficulty="easy",
            expected="answer",
            intent_id="health.doctor",
            forbidden_intents=("recovery.reset",),
            required_sources=(),
        )
        good = CaseResult(case, "answer", "health.doctor", (), 1.0)
        wrong = CaseResult(case, "answer", "recovery.reset", (), 1.0)

        self.assertTrue(case_passes(good))
        self.assertFalse(case_passes(wrong))

    def test_source_case_requires_matching_source_prefix(self):
        case = BenchmarkCase(
            id="source-cli",
            query="where is the CLI dispatcher",
            difficulty="source",
            expected="source",
            intent_id=None,
            forbidden_intents=(),
            required_sources=("ryoku/cli/",),
        )

        self.assertTrue(
            case_passes(
                CaseResult(
                    case,
                    "source",
                    None,
                    ("ryoku/cli/main.go:1-32",),
                    1.0,
                )
            )
        )
        self.assertFalse(
            case_passes(
                CaseResult(
                    case,
                    "source",
                    None,
                    ("docs/cli.md:1-39",),
                    1.0,
                )
            )
        )

    def test_macro_score_weights_tiers_equally(self):
        easy = BenchmarkCase("easy", "q1", "easy", "reject", None, (), ())
        safety = BenchmarkCase(
            "safety", "q2", "safety", "reject", None, (), ()
        )
        report = score_results(
            "hybrid",
            [
                CaseResult(easy, "reject", None, (), 1.0),
                CaseResult(safety, "answer", "recovery.reset", (), 100.0),
            ],
            replay_mismatches=0,
        )

        self.assertEqual(report.tier_scores, {"easy": 1.0, "safety": 0.0})
        self.assertEqual(report.macro_score, 0.5)
        self.assertFalse(report.acceptance_passed)

    def test_latency_summary_uses_nearest_rank_percentiles(self):
        cases = [
            BenchmarkCase(str(i), "q", "easy", "reject", None, (), ())
            for i in range(3)
        ]
        report = score_results(
            "hybrid",
            [
                CaseResult(cases[0], "reject", None, (), 1.0),
                CaseResult(cases[1], "reject", None, (), 2.0),
                CaseResult(cases[2], "reject", None, (), 100.0),
            ],
            replay_mismatches=0,
        )

        self.assertAlmostEqual(report.mean_ms, 103 / 3)
        self.assertEqual(report.p50_ms, 2.0)
        self.assertEqual(report.p95_ms, 100.0)

    def test_load_benchmark_rejects_unknown_difficulty(self):
        document = {
            "schema_version": 1,
            "cases": [
                {
                    "id": "bad",
                    "query": "question",
                    "difficulty": "impossible",
                    "expected": "reject",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "benchmark.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "difficulty"):
                load_benchmark(path)



class FixedRetriever:
    def __init__(self, decision):
        self.decision = decision

    def retrieve(self, query):
        return self.decision


class FixedProwl:
    def __init__(self, result):
        self.result = result
        self.hints = []

    async def search(self, query, source_hints=()):
        self.hints.append(source_hints)
        return self.result


class HybridEvaluatorTests(unittest.IsolatedAsyncioTestCase):
    async def test_maps_curated_answer_to_case_result(self):
        card = SupportCard(
            id="health.doctor",
            title="Doctor",
            examples=("check health", "repair drift"),
            keywords=("doctor",),
            exact_terms=("ryoku doctor",),
            answer="Run doctor.",
            risk="informational",
            docs_url=None,
            source_hints=(),
            clarifies_with=(),
        )
        case = BenchmarkCase(
            "doctor",
            "run doctor",
            "easy",
            "answer",
            "health.doctor",
            (),
            (),
        )
        evaluator = HybridEvaluator(
            FixedRetriever(RetrievalDecision("answer", card=card)),
            None,
        )

        result = await evaluator.evaluate(case)

        self.assertEqual(result.actual, "answer")
        self.assertEqual(result.intent_id, "health.doctor")

    async def test_maps_prowl_hits_to_source_citations(self):
        case = BenchmarkCase(
            "source",
            "where is the CLI implemented",
            "source",
            "source",
            None,
            (),
            ("ryoku/cli/",),
        )
        prowl = FixedProwl(
            ProwlResult(
                "ok",
                hits=(
                    SourceHit(
                        "ryoku/cli/main.go", 1, 32, "package main", False
                    ),
                ),
            )
        )
        card = SupportCard(
            id="overview.repo",
            title="Repository",
            examples=("repository layout", "source tree"),
            keywords=("source",),
            exact_terms=(),
            answer="Repository answer.",
            risk="informational",
            docs_url=None,
            source_hints=("ryoku/cli/main.go",),
            clarifies_with=(),
        )
        evaluator = HybridEvaluator(
            FixedRetriever(RetrievalDecision("answer", card=card)),
            prowl,
        )

        result = await evaluator.evaluate(case)

        self.assertEqual(result.actual, "source")
        self.assertEqual(result.sources, ("ryoku/cli/main.go:1-32",))
        self.assertEqual(prowl.hints, [("ryoku/cli/main.go",)])

if __name__ == "__main__":
    unittest.main()
