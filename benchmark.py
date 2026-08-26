from __future__ import annotations

import argparse
import asyncio
import json
import math
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean

from prowl import ProwlClient, is_source_query
from support import (
    RetrievalDecision,
    SupportRetriever,
    load_support_cards,
    requires_safety_confirmation,
)

DIFFICULTIES = frozenset(
    {
        "easy",
        "paraphrase",
        "contextual",
        "ambiguous",
        "negative",
        "safety",
        "source",
    }
)
OUTCOMES = frozenset({"answer", "clarify", "reject", "source"})
LEGACY_MODEL = "minishlab/potion-code-16M-v2"
HYBRID_MODEL = "minishlab/potion-base-32M"

SOURCE_HIT_GATE = 0.70

LEGACY_TITLE_IDS = {
    "Change the SDDM Theme": "settings.sddm-theme",
    "Check GPG Keys": "packages.gpg-keys",
    "Check Ryoku System Status": "health.status",
    "Check SDDM Logs": "logs.sddm",
    "Check What Ryoku Doctor Would Change": "health.doctor-preview",
    "Check Whether Root Uses Btrfs": "snapshots.root-filesystem",
    "Choose the GPU Ryoku Uses": "hardware.gpu-choice",
    "Create a Ryoku Doctor Report": "health.doctor-report",
    "Detect GPUs in Ryoku": "hardware.gpu-detect",
    "Display Change Was Rejected": "display.rejected-change",
    "Explain Ryoku Problems": "health.doctor-explain",
    "File a Useful Ryoku Bug Report": "support.bug-report",
    "Find Ryoku Logs": "logs.locations",
    "Fix Missing Snapper Root Config": "snapshots.missing-config",
    "Fix a Broken Login Screen": "recovery.login-screen",
    "Fix a Hyprland User Config Error": "shell.hyprland-user-config",
    "Fix a Missing Ryoku Bar or Shell": "shell.missing",
    "Improve Ryoku Performance": "performance.desktop",
    "Install Ryoku on a Dedicated Drive": "install.dedicated-drive",
    "Installer Stuck at Disk Layout": "install.disk-layout",
    "Make Space for Windows Dual Boot": "install.dual-boot-space",
    "Only One Display Resolution Is Available": "display.single-resolution",
    "Recover a Severely Broken Ryoku Install": "recovery.last-resort",
    "Refresh the Ryoku Shell Log": "logs.shell-refresh",
    "Repair Generated Monitor or GPU Config": "display.generated-config",
    "Roll Back Ryoku From a TTY": "recovery.tty-rollback",
    "Roll Back a Bad Ryoku Update": "updates.rollback",
    "Run Ryoku Doctor": "health.doctor",
    "Ryoku Doctor Report Privacy": "health.doctor-report-privacy",
}


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    query: str
    difficulty: str
    expected: str
    intent_id: str | None
    forbidden_intents: tuple[str, ...]
    required_sources: tuple[str, ...]


@dataclass(frozen=True)
class CaseResult:
    case: BenchmarkCase
    actual: str
    intent_id: str | None
    sources: tuple[str, ...]
    latency_ms: float


@dataclass(frozen=True)
class BenchmarkReport:
    mode: str
    case_count: int
    passed: int
    tier_scores: dict[str, float]
    macro_score: float
    answer_accuracy: float
    clarification_accuracy: float
    negative_rejection_rate: float
    safety_pass_rate: float
    source_hit_rate: float
    mean_ms: float
    p50_ms: float
    p95_ms: float
    replay_mismatches: int
    acceptance_passed: bool


def _strict_keys(value: dict, allowed: set[str], context: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{context} has unknown fields: {', '.join(unknown)}")


def load_benchmark(path: Path) -> list[BenchmarkCase]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("benchmark must be a JSON object")
    _strict_keys(raw, {"schema_version", "cases"}, "benchmark")
    if raw.get("schema_version") != 1:
        raise ValueError("benchmark schema_version must be 1")
    rows = raw.get("cases")
    if not isinstance(rows, list) or not rows:
        raise ValueError("benchmark cases must be a non-empty list")

    cases: list[BenchmarkCase] = []
    seen: set[str] = set()
    allowed = {
        "id",
        "query",
        "difficulty",
        "expected",
        "intent_id",
        "forbidden_intents",
        "required_sources",
    }
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise TypeError(f"case #{index} must be an object")
        _strict_keys(row, allowed, f"case #{index}")
        case_id = row.get("id")
        query = row.get("query")
        difficulty = row.get("difficulty")
        expected = row.get("expected")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"case #{index} id must be a non-empty string")
        if case_id in seen:
            raise ValueError(f"duplicate benchmark id: {case_id}")
        seen.add(case_id)
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"case {case_id} query must be a non-empty string")
        if difficulty not in DIFFICULTIES:
            raise ValueError(f"case {case_id} has invalid difficulty: {difficulty}")
        if expected not in OUTCOMES:
            raise ValueError(f"case {case_id} has invalid expected outcome: {expected}")
        intent_id = row.get("intent_id")
        if intent_id is not None and not isinstance(intent_id, str):
            raise ValueError(f"case {case_id} intent_id must be a string or null")
        forbidden = row.get("forbidden_intents", [])
        sources = row.get("required_sources", [])
        if not isinstance(forbidden, list) or not all(
            isinstance(value, str) and value for value in forbidden
        ):
            raise ValueError(f"case {case_id} forbidden_intents must be strings")
        if not isinstance(sources, list) or not all(
            isinstance(value, str) and value for value in sources
        ):
            raise ValueError(f"case {case_id} required_sources must be strings")
        cases.append(
            BenchmarkCase(
                id=case_id,
                query=query.strip(),
                difficulty=difficulty,
                expected=expected,
                intent_id=intent_id,
                forbidden_intents=tuple(forbidden),
                required_sources=tuple(sources),
            )
        )
    return cases


def case_passes(result: CaseResult) -> bool:
    case = result.case
    if result.actual != case.expected:
        return False
    if case.intent_id is not None and result.intent_id != case.intent_id:
        return False
    if result.intent_id in case.forbidden_intents:
        return False
    return all(
        any(source.startswith(prefix) for source in result.sources)
        for prefix in case.required_sources
    )


def _rate(results: Sequence[CaseResult], predicate) -> float:
    selected = [result for result in results if predicate(result.case)]
    if not selected:
        return 1.0
    return sum(case_passes(result) for result in selected) / len(selected)


def _nearest_rank(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def score_results(
    mode: str,
    results: Sequence[CaseResult],
    replay_mismatches: int,
) -> BenchmarkReport:
    if not results:
        raise ValueError("cannot score an empty benchmark")
    grouped: dict[str, list[CaseResult]] = {}
    for result in results:
        grouped.setdefault(result.case.difficulty, []).append(result)
    tier_scores = {
        tier: sum(case_passes(result) for result in rows) / len(rows)
        for tier, rows in sorted(grouped.items())
    }
    latencies = [result.latency_ms for result in results]
    answer_accuracy = _rate(results, lambda case: case.expected == "answer")
    clarification_accuracy = _rate(
        results, lambda case: case.expected == "clarify"
    )
    negative_rejection_rate = _rate(
        results, lambda case: case.difficulty == "negative"
    )
    safety_pass_rate = _rate(
        results, lambda case: case.difficulty == "safety"
    )
    source_hit_rate = _rate(results, lambda case: case.expected == "source")
    acceptance = (
        answer_accuracy >= 0.90
        and clarification_accuracy == 1.0
        and negative_rejection_rate >= 0.95
        and safety_pass_rate == 1.0
        and source_hit_rate >= SOURCE_HIT_GATE
        and replay_mismatches == 0
    )
    return BenchmarkReport(
        mode=mode,
        case_count=len(results),
        passed=sum(case_passes(result) for result in results),
        tier_scores=tier_scores,
        macro_score=fmean(tier_scores.values()),
        answer_accuracy=answer_accuracy,
        clarification_accuracy=clarification_accuracy,
        negative_rejection_rate=negative_rejection_rate,
        safety_pass_rate=safety_pass_rate,
        source_hit_rate=source_hit_rate,
        mean_ms=fmean(latencies),
        p50_ms=_nearest_rank(latencies, 0.50),
        p95_ms=_nearest_rank(latencies, 0.95),
        replay_mismatches=replay_mismatches,
        acceptance_passed=acceptance,
    )


def _normalize_rows(vectors):
    import numpy as np

    values = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.clip(norms, 1e-12, None)


class LegacyEvaluator:
    def __init__(self, faq_path: Path, encoder, threshold: float):
        raw = json.loads(faq_path.read_text(encoding="utf-8"))
        if not isinstance(raw, list) or not raw:
            raise ValueError("legacy FAQ must be a non-empty list")
        self._entries = raw
        self._encoder = encoder
        self._threshold = threshold
        self._vectors = _normalize_rows(
            encoder.encode([entry["text"] for entry in raw])
        )

    def evaluate(self, case: BenchmarkCase) -> CaseResult:
        import numpy as np

        started = time.perf_counter()
        query_vector = _normalize_rows(self._encoder.encode([case.query]))[0]
        scores = self._vectors @ query_vector
        index = int(np.argmax(scores))
        score = float(scores[index])
        if score < self._threshold:
            actual = "reject"
            intent_id = None
        else:
            actual = "answer"
            intent_id = LEGACY_TITLE_IDS.get(self._entries[index]["title"])
        return CaseResult(
            case=case,
            actual=actual,
            intent_id=intent_id,
            sources=(),
            latency_ms=(time.perf_counter() - started) * 1000,
        )


class HybridEvaluator:
    def __init__(
        self,
        retriever: SupportRetriever,
        prowl: ProwlClient | None,
    ):
        self.retriever = retriever
        self.prowl = prowl

    async def evaluate(self, case: BenchmarkCase) -> CaseResult:
        started = time.perf_counter()
        decision: RetrievalDecision = self.retriever.retrieve(case.query)
        actual = "reject"
        intent_id = None
        sources: tuple[str, ...] = ()
        dangerous = requires_safety_confirmation(case.query)
        source_query = is_source_query(case.query)
        if dangerous:
            actual = "reject" if source_query else "clarify"
        elif self.prowl is not None and source_query:
            source_hints = (
                decision.card.source_hints
                if decision.kind == "answer" and decision.card
                else ()
            )
            result = await self.prowl.search(case.query, source_hints)
            if result.status == "ok":
                actual = "source"
                sources = tuple(hit.citation for hit in result.hits)
        elif decision.kind == "answer" and decision.card is not None:
            actual = "answer"
            intent_id = decision.card.id
        elif decision.kind == "clarify":
            actual = "clarify"
        return CaseResult(
            case=case,
            actual=actual,
            intent_id=intent_id,
            sources=sources,
            latency_ms=(time.perf_counter() - started) * 1000,
        )


async def evaluate_hybrid(
    evaluator: HybridEvaluator,
    cases: Sequence[BenchmarkCase],
    replays: int,
) -> tuple[list[CaseResult], int]:
    results: list[CaseResult] = []
    mismatches = 0
    for case in cases:
        first = await evaluator.evaluate(case)
        results.append(first)
        signature = (first.actual, first.intent_id, first.sources)
        for _ in range(max(1, replays) - 1):
            replay = await evaluator.evaluate(case)
            if (replay.actual, replay.intent_id, replay.sources) != signature:
                mismatches += 1
    return results, mismatches


def _report_dict(
    report: BenchmarkReport, results: Sequence[CaseResult]
) -> dict:
    payload = asdict(report)
    payload["results"] = [
        {
            "id": result.case.id,
            "difficulty": result.case.difficulty,
            "expected": result.case.expected,
            "actual": result.actual,
            "expected_intent": result.case.intent_id,
            "actual_intent": result.intent_id,
            "sources": list(result.sources),
            "latency_ms": result.latency_ms,
            "passed": case_passes(result),
        }
        for result in results
    ]
    return payload


def _print_human(report: BenchmarkReport) -> None:
    print(
        f"{report.mode}: {report.passed}/{report.case_count} cases "
        f"macro={report.macro_score:.1%}"
    )
    for tier, score in report.tier_scores.items():
        print(f"  {tier:10s} {score:.1%}")
    print(
        f"  latency mean={report.mean_ms:.2f}ms "
        f"p50={report.p50_ms:.2f}ms p95={report.p95_ms:.2f}ms"
    )
    print(
        f"  safety={report.safety_pass_rate:.1%} "
        f"negative={report.negative_rejection_rate:.1%} "
        f"source={report.source_hit_rate:.1%} "
        f"determinism_mismatches={report.replay_mismatches}"
    )
    print(f"  acceptance={'PASS' if report.acceptance_passed else 'FAIL'}")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("legacy", "hybrid"), default="legacy"
    )
    parser.add_argument(
        "--benchmark", type=Path, default=Path("data/benchmark.json")
    )
    parser.add_argument("--faq", type=Path, default=Path("data/faq.json"))
    parser.add_argument(
        "--support", type=Path, default=Path("data/support.json")
    )
    parser.add_argument("--ryoku-repo", type=Path)
    parser.add_argument("--prowl-smart", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--replays", type=int, default=1)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)

    from model2vec import StaticModel

    cases = load_benchmark(args.benchmark)
    model_name = args.model or (
        LEGACY_MODEL if args.mode == "legacy" else HYBRID_MODEL
    )
    model = StaticModel.from_pretrained(model_name)
    if args.mode == "legacy":
        evaluator = LegacyEvaluator(args.faq, model, args.threshold)
        results = [evaluator.evaluate(case) for case in cases]
        mismatches = 0
    else:
        retriever = SupportRetriever(load_support_cards(args.support), model)
        prowl = (
            ProwlClient(args.ryoku_repo, smart_search=args.prowl_smart)
            if args.ryoku_repo
            else None
        )
        results, mismatches = asyncio.run(
            evaluate_hybrid(
                HybridEvaluator(retriever, prowl),
                cases,
                max(1, args.replays),
            )
        )
    report = score_results(args.mode, results, mismatches)
    if args.json:
        print(json.dumps(_report_dict(report, results), indent=2, sort_keys=True))
    else:
        _print_human(report)
    return 1 if args.check and not report.acceptance_passed else 0


if __name__ == "__main__":
    raise SystemExit(main())
