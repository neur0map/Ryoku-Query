import tempfile
import unittest
from pathlib import Path

from archive_review import ArchiveReviewStore


class ArchiveReviewStoreTests(unittest.TestCase):
    def test_imports_identifier_only_pairs_and_deduplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ArchiveReviewStore(Path(directory) / "archive.sqlite3")
            records = [
                {
                    "request_message_id": "101",
                    "answer_message_id": "102",
                    "requester_id": "10",
                    "bot_id": "20",
                    "channel_id": "30",
                    "created_at": "2026-08-27T12:00:00+00:00",
                    "link_method": "reply",
                    "topic": "ryoku",
                }
            ]
            self.assertEqual(store.import_pairs(records, archive_sha256="a" * 64), 1)
            self.assertEqual(store.import_pairs(records, archive_sha256="a" * 64), 0)
            self.assertEqual(store.counts(), {"ryoku": 1})
            self.assertEqual(store.pending_source_reviews(), [102])

    def test_rejects_content_bearing_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ArchiveReviewStore(Path(directory) / "archive.sqlite3")
            with self.assertRaises(ValueError):
                store.import_pairs(
                    [{
                        "request_message_id": "101", "answer_message_id": "102",
                        "requester_id": "10", "bot_id": "20", "channel_id": "30",
                        "created_at": None, "link_method": "reply",
                        "topic": "the archived question text",
                    }],
                    archive_sha256="archive-sha",
                )
            store.import_pairs(
                [{
                    "request_message_id": "101", "answer_message_id": "102",
                    "requester_id": "10", "bot_id": "20", "channel_id": "30",
                    "created_at": None, "link_method": "reply", "topic": "ryoku",
                }],
                archive_sha256="a" * 64,
            )
            with self.assertRaises(ValueError):
                store.record_source_review(
                    102,
                    verdict="source_verified",
                    source_revision="b" * 40,
                    evidence_paths=["historical answer text\nleak"],
                )

    def test_records_source_verification_without_message_text(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ArchiveReviewStore(Path(directory) / "archive.sqlite3")
            store.import_pairs(
                [{
                    "request_message_id": "101", "answer_message_id": "102",
                    "requester_id": "10", "bot_id": "20", "channel_id": "30",
                    "created_at": None, "link_method": "reply", "topic": "ryoku",
                }],
                archive_sha256="a" * 64,
            )
            store.record_source_review(
                102,
                verdict="source_verified",
                source_revision="b" * 40,
                evidence_paths=["docs/updates.md:1-12"],
            )
            review = store.review_for(102)
            self.assertEqual(review["verdict"], "source_verified")
            self.assertEqual(review["evidence_paths"], ["docs/updates.md:1-12"])
            self.assertNotIn("question", review)
            self.assertNotIn("answer", review)


if __name__ == "__main__":
    unittest.main()
