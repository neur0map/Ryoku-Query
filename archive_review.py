"""Identifier-only audit storage for approved historical Discord archives.

This is deliberately separate from live feedback and never stores question or
answer content. Archive message IDs remain join keys to the protected archive.
"""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any


class ArchiveReviewStore:
    """Store message identifiers and bounded source-review metadata only."""

    _ID = re.compile(r"^[1-9][0-9]{0,19}$")
    _SHA256 = re.compile(r"^[0-9a-f]{64}$")
    _GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
    _EVIDENCE = re.compile(r"^[A-Za-z0-9_./-]+:[1-9][0-9]*(?:-[1-9][0-9]*)?$")
    _TOPICS = {"other", "ryoku"}

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS archive_pairs (
                    answer_message_id INTEGER PRIMARY KEY,
                    request_message_id INTEGER NOT NULL,
                    requester_id INTEGER NOT NULL,
                    bot_id INTEGER NOT NULL,
                    channel_id INTEGER NOT NULL,
                    created_at TEXT,
                    link_method TEXT NOT NULL CHECK (link_method IN ('reply', 'nearby')),
                    topic TEXT NOT NULL,
                    archive_sha256 TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS source_reviews (
                    answer_message_id INTEGER PRIMARY KEY,
                    verdict TEXT NOT NULL CHECK (verdict IN (
                        'pending_source_review', 'source_verified', 'incorrect', 'insufficient_evidence'
                    )),
                    source_revision TEXT,
                    evidence_paths_json TEXT NOT NULL DEFAULT '[]',
                    FOREIGN KEY (answer_message_id) REFERENCES archive_pairs(answer_message_id)
                );
                CREATE INDEX IF NOT EXISTS archive_pairs_topic_idx ON archive_pairs(topic);
                CREATE INDEX IF NOT EXISTS source_reviews_verdict_idx ON source_reviews(verdict);
                """
            )

    def import_pairs(self, records: list[dict[str, Any]], *, archive_sha256: str) -> int:
        self._validate_archive_sha256(archive_sha256)
        for record in records:
            self._validate_pair(record)
        inserted = 0
        with self._connect() as connection:
            for record in records:
                before = connection.execute(
                    """
                    INSERT OR IGNORE INTO archive_pairs (
                        answer_message_id, request_message_id, requester_id, bot_id,
                        channel_id, created_at, link_method, topic, archive_sha256
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        int(record["answer_message_id"]),
                        int(record["request_message_id"]),
                        int(record["requester_id"]),
                        int(record["bot_id"]),
                        int(record["channel_id"]),
                        record.get("created_at"),
                        record["link_method"],
                        record["topic"],
                        archive_sha256,
                    ),
                )
                if before.rowcount:
                    inserted += 1
                    connection.execute(
                        """
                        INSERT INTO source_reviews (answer_message_id, verdict)
                        VALUES (?, 'pending_source_review')
                        """,
                        (int(record["answer_message_id"]),),
                    )
        return inserted

    def record_source_review(
        self,
        answer_message_id: int,
        *,
        verdict: str,
        source_revision: str | None,
        evidence_paths: list[str],
    ) -> None:
        if verdict not in {
            "pending_source_review", "source_verified", "incorrect", "insufficient_evidence"
        }:
            raise ValueError("invalid source-review verdict")
        self._validate_review_metadata(source_revision, evidence_paths)
        with self._connect() as connection:
            if connection.execute(
                "SELECT 1 FROM archive_pairs WHERE answer_message_id = ?",
                (answer_message_id,),
            ).fetchone() is None:
                raise ValueError("unknown archive answer")
            connection.execute(
                """
                INSERT INTO source_reviews (
                    answer_message_id, verdict, source_revision, evidence_paths_json
                ) VALUES (?, ?, ?, ?)
                ON CONFLICT(answer_message_id) DO UPDATE SET
                    verdict = excluded.verdict,
                    source_revision = excluded.source_revision,
                    evidence_paths_json = excluded.evidence_paths_json
                """,
                (answer_message_id, verdict, source_revision, json.dumps(evidence_paths)),
            )

    def pending_source_reviews(self) -> list[int]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT archive_pairs.answer_message_id
                FROM archive_pairs JOIN source_reviews USING (answer_message_id)
                WHERE archive_pairs.topic = 'ryoku'
                  AND source_reviews.verdict = 'pending_source_review'
                ORDER BY archive_pairs.answer_message_id
                """
            ).fetchall()
        return [row[0] for row in rows]

    def review_for(self, answer_message_id: int) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT verdict, source_revision, evidence_paths_json
                FROM source_reviews WHERE answer_message_id = ?
                """,
                (answer_message_id,),
            ).fetchone()
        if row is None:
            raise ValueError("unknown archive answer")
        return {"verdict": row[0], "source_revision": row[1], "evidence_paths": json.loads(row[2])}

    def counts(self) -> dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT topic, COUNT(*) FROM archive_pairs GROUP BY topic ORDER BY topic"
            ).fetchall()
        return {topic: count for topic, count in rows}

    def _validate_archive_sha256(self, value: str) -> None:
        if not isinstance(value, str) or self._SHA256.fullmatch(value) is None:
            raise ValueError("archive_sha256 must be a lowercase SHA-256")

    def _validate_pair(self, record: dict[str, Any]) -> None:
        for field in ("request_message_id", "answer_message_id", "requester_id", "bot_id", "channel_id"):
            value = str(record.get(field, ""))
            if self._ID.fullmatch(value) is None:
                raise ValueError(f"{field} must be a positive Discord snowflake")
        if record.get("link_method") not in {"reply", "nearby"}:
            raise ValueError("invalid link_method")
        if record.get("topic") not in self._TOPICS:
            raise ValueError("topic must be an allowed classification")
        created_at = record.get("created_at")
        if created_at is not None and (
            not isinstance(created_at, str)
            or len(created_at) > 35
            or "\n" in created_at
            or "T" not in created_at
        ):
            raise ValueError("created_at must be an ISO timestamp or null")

    def _validate_review_metadata(
        self, source_revision: str | None, evidence_paths: list[str]
    ) -> None:
        if source_revision is not None and (
            not isinstance(source_revision, str)
            or self._GIT_SHA.fullmatch(source_revision) is None
        ):
            raise ValueError("source_revision must be a full lowercase Git SHA or null")
        if not isinstance(evidence_paths, list) or len(evidence_paths) > 32:
            raise ValueError("evidence_paths must be a bounded list")
        if any(
            not isinstance(path, str) or self._EVIDENCE.fullmatch(path) is None
            for path in evidence_paths
        ):
            raise ValueError("evidence paths must be source path and line range references")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.execute("PRAGMA foreign_keys = ON")
        return connection
