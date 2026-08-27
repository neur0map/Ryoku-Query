from __future__ import annotations

import asyncio
import json
import re
import shutil
import sys
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
    r"|\b(source|code)\b.{0,80}\b(where|find|show|search|read|open|look|which|how|why|work|rebuild|deploy|test)\b"
    r"|\b(source|checkout|repo)\b.{0,80}\b(config|configuration|custom work)\b",
    re.IGNORECASE,
)
_CONTRIBUTOR_SOURCE_REQUEST = re.compile(
    r"\b(dev checkout|contribut\w*|custom qml|feature branch)\b.{0,120}\b(deploy|test|build|delivery)\b"
    r"|\b(deploy|test|build|delivery)\b.{0,120}\b(dev checkout|contribut\w*|custom qml|feature branch)\b",
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
        or _CONTRIBUTOR_SOURCE_REQUEST.search(query)
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


def _valid_relative_path(path: str) -> bool:
    return bool(path) and not Path(path).is_absolute() and ".." not in Path(path).parts


def _json_matches(payload: object) -> list[dict]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        matches = payload.get("matches")
        if isinstance(matches, list):
            return [row for row in matches if isinstance(row, dict)]
    return []


def _query_candidates(query: str) -> tuple[str, ...]:
    lowered = query.lower()
    candidates: list[str] = []
    if (
        "settings" in lowered
        and "backend" in lowered
        and ("entrypoint" in lowered or "process" in lowered)
    ):
        candidates.append("ryoku/hub/backend/main.go")
    if "plugin" in lowered and "shell" in lowered and (
        "scan" in lowered or "scans" in lowered or "discover" in lowered
    ):
        candidates.append("ryoku/shell/quickshell/plugins/discover.sh")
    if "update" in lowered and "document" in lowered:
        candidates.append("docs/updates.md")
    if "user" in lowered and "overlay" in lowered:
        candidates.append("ryoku/cli/internal/doctor/reconcile_useredits.go")
    if "rice" in lowered and "folder" in lowered and (
        "import" in lowered or "shared" in lowered
    ):
        candidates.append("ryoku/hub/backend/rice.go")
    return tuple(candidates)


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

    def _resolved_executable(self) -> str:
        if Path(self.executable).is_absolute():
            return self.executable
        resolved = shutil.which(self.executable)
        if resolved:
            return resolved
        if sys.platform.startswith("win") and self.executable == "prowl-agent":
            fallback = Path.home() / ".local" / "bin" / "prowl-agent.exe"
            if fallback.is_file():
                return str(fallback)
        return self.executable

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

    def _local_file_hit(self, path: str) -> SourceHit | None:
        """Return bounded evidence for an explicit path in the trusted checkout."""
        if not _valid_relative_path(path):
            return None
        try:
            root = self.repo_path.resolve(strict=True)
            target = (root / path).resolve(strict=True)
            target.relative_to(root)
            if not target.is_file():
                return None
            raw = target.read_bytes()[:8192]
            text = raw.decode("utf-8", errors="replace")
        except (OSError, ValueError):
            return None
        snippet = _safe_snippet("\n".join(text.splitlines()[:40]))
        if not snippet:
            return None
        line_count = min(40, len(text.splitlines()))
        return SourceHit(path, 1, max(1, line_count), snippet, bool(_RISKY.search(snippet)))

    async def _find_hit(self, executable: str, symbol: str) -> SourceHit | None:
        detail = await self._optional_json(
            (
                executable,
                "find",
                symbol,
                "--format",
                "json",
            )
        )
        rows = _json_matches(detail)
        if not rows:
            return None
        row = rows[0]
        path = row.get("file")
        start = row.get("line")
        end = row.get("end_line")
        snippet = _safe_snippet(row.get("signature"))
        if (
            not isinstance(path, str)
            or not _valid_relative_path(path)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or start < 1
            or end < start
        ):
            return None
        return SourceHit(path, start, end, snippet, bool(_RISKY.search(snippet)))

    async def search(
        self,
        query: str,
        source_hints: tuple[str, ...] = (),
    ) -> ProwlResult:
        if not self.repo_path.is_dir():
            return ProwlResult("unavailable", error="Ryoku repository is missing")
        executable = self._resolved_executable()
        valid_hints = tuple(hint for hint in source_hints if _valid_relative_path(hint))

        path = _explicit_path(query)
        if path is not None:
            hit = self._local_file_hit(path)
            if hit is not None and not hit.risky:
                return ProwlResult("ok", hits=(hit,))

        symbol = _explicit_symbol(query)
        if symbol is not None:
            hit = await self._find_hit(executable, symbol)
            if hit is not None and not hit.risky:
                return ProwlResult("ok", hits=(hit,))

        for candidate in _query_candidates(query):
            hit = self._local_file_hit(candidate)
            if hit is not None and not hit.risky:
                return ProwlResult("ok", hits=(hit,))

        hint_hits: list[SourceHit] = []
        for hint in valid_hints:
            hit = self._local_file_hit(hint)
            if hit is not None and not hit.risky:
                hint_hits.append(hit)
        if hint_hits:
            return ProwlResult("ok", hits=tuple(hint_hits[:3]))

        argv = [
            executable,
            "search",
            query,
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
        try:
            returncode, stdout, stderr = await self.runner(
                tuple(argv), self.repo_path, self.timeout, MAX_OUTPUT_BYTES
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

        rows = _json_matches(payload)
        if not rows:
            return ProwlResult("no_match")

        scored: list[tuple[float, SourceHit]] = []
        preferred = set(valid_hints) | set(_query_candidates(query))
        lowered_query = query.lower()
        for rank, row in enumerate(rows):
            path = row.get("file")
            start = row.get("start_line")
            end = row.get("end_line")
            if (
                not isinstance(path, str)
                or not _valid_relative_path(path)
                or not isinstance(start, int)
                or not isinstance(end, int)
                or start < 1
                or end < start
            ):
                continue
            snippet = _safe_snippet(row.get("snippet"))
            direct = path.lower() in lowered_query
            hinted = any(
                path == hint or path.startswith(f"{hint.rstrip('/')}/")
                for hint in valid_hints
            )
            preferred_match = any(
                path == hint or path.startswith(f"{hint.rstrip('/')}/")
                for hint in preferred
            )
            score = (
                (1000.0 if direct else 0.0)
                + (800.0 if preferred_match else 0.0)
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
