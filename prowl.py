from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

MAX_OUTPUT_BYTES = 256 * 1024
_SOURCE_WORDS = re.compile(
    r"\b(implemented|implementation|function|class|symbol|defined|definition)\b",
    re.IGNORECASE,
)
_SOURCE_REQUEST = re.compile(
    r"\b(where|find|show|search|read|open|look|which)\b.{0,80}\b(source|code)\b"
    r"|\b(source|code)\b.{0,80}\b(where|find|show|search|read|open|look|which)\b",
    re.IGNORECASE,
)
_SOURCE_PATH = re.compile(
    r"\b(?:[\w.-]+/)*[\w.-]+\.(?:go|py|qml|lua|sh|md|json|toml|yaml)\b",
    re.IGNORECASE,
)
_SOURCE_SYMBOL = re.compile(
    r"`([A-Za-z_][A-Za-z0-9_]*)`|\b([a-z]+[A-Z][A-Za-z0-9_]*)\b"
)
_RISKY = re.compile(
    r"\b(?:rm\s+-rf|mkfs|wipefs|repartition|erase\s+whole\s+disk|ryoku\s+recovery)\b",
    re.IGNORECASE,
)
Runner = Callable[
    [tuple[str, ...], Path, float, int],
    Awaitable[tuple[int, bytes, bytes]],
]


@dataclass(frozen=True)
class SourceHit:
    file: str
    start_line: int
    end_line: int
    snippet: str
    risky: bool

    @property
    def citation(self) -> str:
        return f"{self.file}:{self.start_line}-{self.end_line}"


@dataclass(frozen=True)
class ProwlResult:
    status: str
    hits: tuple[SourceHit, ...] = ()
    error: str | None = None


def is_source_query(query: str) -> bool:
    return bool(
        _SOURCE_WORDS.search(query)
        or _SOURCE_REQUEST.search(query)
        or _SOURCE_PATH.search(query)
        or _SOURCE_SYMBOL.search(query)
    )


async def _run(
    argv: tuple[str, ...], cwd: Path, timeout: float, max_output: int
) -> tuple[int, bytes, bytes]:
    process = await asyncio.create_subprocess_exec(
        *argv,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(), timeout=timeout
        )
    except TimeoutError:
        process.kill()
        await process.wait()
        raise
    if len(stdout) + len(stderr) > max_output:
        raise ValueError("Prowl output exceeded the configured bound")
    return process.returncode, stdout, stderr


def _explicit_path(query: str) -> str | None:
    match = _SOURCE_PATH.search(query)
    return match.group(0) if match else None


def _explicit_symbol(query: str) -> str | None:
    match = _SOURCE_SYMBOL.search(query)
    if not match:
        return None
    return match.group(1) or match.group(2)


def _path_weight(path: str) -> float:
    lowered = path.lower()
    if lowered.startswith("docs/") or lowered == "readme.md":
        return 4.0
    if lowered.startswith("ryoku/cli/"):
        return 3.0
    if lowered.startswith("installation/") and "readme" in lowered:
        return 2.5
    if lowered.startswith((".github/", "tests/")) or "changelog" in lowered:
        return 0.3
    return 1.0


def _safe_snippet(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value[:800].replace("@", "@\u200b")


class ProwlClient:
    def __init__(
        self,
        repo_path: Path,
        timeout: float = 4.0,
        result_limit: int = 20,
        executable: str = "prowl-agent",
        smart_search: bool = False,
        runner: Runner | None = None,
    ):
        self.repo_path = repo_path
        self.timeout = max(0.1, min(float(timeout), 30.0))
        self.result_limit = max(1, min(int(result_limit), 20))
        self.executable = executable
        self.smart_search = smart_search
        self.runner = runner or _run

    async def _optional_json(self, argv: tuple[str, ...]):
        try:
            returncode, stdout, _ = await self.runner(
                argv, self.repo_path, self.timeout, MAX_OUTPUT_BYTES
            )
            return json.loads(stdout) if returncode == 0 else None
        except (
            FileNotFoundError,
            OSError,
            TimeoutError,
            UnicodeDecodeError,
            ValueError,
            json.JSONDecodeError,
        ):
            return None

    async def search(
        self,
        query: str,
        source_hints: tuple[str, ...] = (),
    ) -> ProwlResult:
        if not self.repo_path.is_dir():
            return ProwlResult("unavailable", error="Ryoku repository is missing")
        search_query = query
        argv = [
            self.executable,
            "search",
            search_query,
        ]
        if self.smart_search:
            argv.append("--smart")
        argv.extend(
            (
                "--format",
                "json",
                "--limit",
                str(self.result_limit),
            )
        )
        argv = tuple(argv)
        try:
            returncode, stdout, stderr = await self.runner(
                argv, self.repo_path, self.timeout, MAX_OUTPUT_BYTES
            )
        except TimeoutError:
            return ProwlResult("unavailable", error="Prowl search timed out")
        except FileNotFoundError:
            return ProwlResult("unavailable", error="prowl-agent is not installed")
        except ValueError as error:
            return ProwlResult("unavailable", error=str(error))
        except OSError as error:
            return ProwlResult("unavailable", error=f"Prowl search failed: {error}")
        if returncode != 0:
            detail = _safe_snippet(stderr.decode("utf-8", errors="replace"))
            return ProwlResult(
                "unavailable", error=detail or "Prowl search failed"
            )
        try:
            payload = json.loads(stdout)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return ProwlResult("unavailable", error="Prowl returned invalid JSON")
        if not isinstance(payload, list):
            return ProwlResult("unavailable", error="Prowl returned invalid JSON")

        scored: list[tuple[float, SourceHit]] = []
        valid_hints = tuple(
            hint
            for hint in source_hints
            if not Path(hint).is_absolute()
            and ".." not in Path(hint).parts
        )
        path = _explicit_path(query)
        symbol = _explicit_symbol(query)
        if path is not None:
            detail = await self._optional_json(
                (
                    self.executable,
                    "peek",
                    f"{path}:1-40",
                    "--format",
                    "json",
                )
            )
            if isinstance(detail, dict):
                start = detail.get("start_line")
                end = detail.get("end_line")
                text = _safe_snippet(detail.get("text"))
                if isinstance(start, int) and isinstance(end, int):
                    scored.append(
                        (
                            3000.0,
                            SourceHit(
                                path,
                                start,
                                end,
                                text,
                                bool(_RISKY.search(text)),
                            ),
                        )
                    )
        if symbol is not None:
            detail = await self._optional_json(
                (
                    self.executable,
                    "find",
                    symbol,
                    "--format",
                    "json",
                )
            )
            if isinstance(detail, list) and detail:
                row = detail[0]
                path = row.get("file") if isinstance(row, dict) else None
                start = row.get("line") if isinstance(row, dict) else None
                end = row.get("end_line") if isinstance(row, dict) else None
                snippet = _safe_snippet(
                    row.get("signature") if isinstance(row, dict) else ""
                )
                if (
                    isinstance(path, str)
                    and isinstance(start, int)
                    and isinstance(end, int)
                ):
                    scored.append(
                        (
                            3000.0,
                            SourceHit(
                                path,
                                start,
                                end,
                                snippet,
                                bool(_RISKY.search(snippet)),
                            ),
                        )
                    )
        lowered_query = query.lower()
        for rank, row in enumerate(payload):
            if not isinstance(row, dict):
                continue
            path = row.get("file")
            start = row.get("start_line")
            end = row.get("end_line")
            if (
                not isinstance(path, str)
                or not path
                or not isinstance(start, int)
                or not isinstance(end, int)
                or start < 1
                or end < start
                or Path(path).is_absolute()
                or ".." in Path(path).parts
            ):
                continue
            snippet = _safe_snippet(row.get("snippet"))
            direct = path.lower() in lowered_query
            hinted = any(
                path == hint or path.startswith(f"{hint.rstrip('/')}/")
                for hint in valid_hints
            )
            score = (
                (1000.0 if direct else 0.0)
                + (500.0 if hinted else 0.0)
                + _path_weight(path) * (self.result_limit - rank)
            )
            scored.append(
                (
                    score,
                    SourceHit(
                        file=path,
                        start_line=start,
                        end_line=end,
                        snippet=snippet,
                        risky=bool(_RISKY.search(snippet)),
                    ),
                )
            )
        scored.sort(key=lambda item: (-item[0], item[1].citation))
        hits: list[SourceHit] = []
        seen: set[str] = set()
        for _, hit in scored:
            if hit.risky or hit.citation in seen:
                continue
            seen.add(hit.citation)
            hits.append(hit)
            if len(hits) == 3:
                break
        return (
            ProwlResult("ok", hits=tuple(hits))
            if hits
            else ProwlResult("no_match")
        )
