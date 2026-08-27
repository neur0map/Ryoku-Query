import tempfile
import unittest
from pathlib import Path

from feedback import FeedbackStore, FeedbackView


class FeedbackStoreTests(unittest.TestCase):
    def test_records_answer_feedback_and_returns_it_for_later_export(self):
        with tempfile.TemporaryDirectory() as directory:
            store = FeedbackStore(Path(directory) / "nero-feedback.sqlite3")
            store.record_answer(
                answer_message_id=101,
                request_message_id=100,
                requester_id=7,
            )

            outcome = store.record_feedback(
                answer_message_id=101,
                actor_id=7,
                verdict="correct",
            )

            self.assertEqual(outcome, "recorded")
            self.assertEqual(
                store.recent_feedback(),
                [
                    {
                        "answer_message_id": 101,
                        "request_message_id": 100,
                        "requester_id": 7,
                        "actor_id": 7,
                        "verdict": "correct",
                    }
                ],
            )

    def test_replaces_one_requesters_verdict_without_creating_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            store = FeedbackStore(Path(directory) / "nero-feedback.sqlite3")
            store.record_answer(
                answer_message_id=101,
                request_message_id=100,
                requester_id=7,
            )
            store.record_feedback(101, 7, "incorrect")

            outcome = store.record_feedback(101, 7, "correct")

            self.assertEqual(outcome, "updated")
            self.assertEqual(len(store.recent_feedback()), 1)
            self.assertEqual(store.recent_feedback()[0]["verdict"], "correct")

    def test_rejects_feedback_from_someone_other_than_the_requester(self):
        with tempfile.TemporaryDirectory() as directory:
            store = FeedbackStore(Path(directory) / "nero-feedback.sqlite3")
            store.record_answer(
                answer_message_id=101,
                request_message_id=100,
                requester_id=7,
            )

            outcome = store.record_feedback(101, 8, "incorrect")

            self.assertEqual(outcome, "not_requester")
            self.assertEqual(store.recent_feedback(), [])

    def test_view_offers_correct_and_incorrect_buttons(self):
        with tempfile.TemporaryDirectory() as directory:
            view = FeedbackView(FeedbackStore(Path(directory) / "nero-feedback.sqlite3"))

        self.assertEqual([item.label for item in view.children], ["Correct", "Incorrect"])
        self.assertEqual(
            [item.custom_id for item in view.children],
            ["nero_feedback:correct", "nero_feedback:incorrect"],
        )


if __name__ == "__main__":
    unittest.main()
