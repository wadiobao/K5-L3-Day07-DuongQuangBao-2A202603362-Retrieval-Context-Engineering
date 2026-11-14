import json
import unittest
from types import SimpleNamespace

from src.judge_suggestions import _parse_scores


class JudgeSuggestionTests(unittest.TestCase):
    def test_parses_valid_draft_scores(self) -> None:
        response = SimpleNamespace(
            text=json.dumps(
                {
                    "scores": [
                        {
                            "case_id": "case-1",
                            "score": 2,
                            "rationale": "The chunk is self-contained.",
                            "confidence": "high",
                        }
                    ]
                }
            )
        )

        result = _parse_scores(response, {"case-1"})

        self.assertEqual(result["case-1"]["score"], 2)
        self.assertEqual(result["case-1"]["confidence"], "high")

    def test_parses_json_fence_without_changing_score_meaning(self) -> None:
        response = SimpleNamespace(
            text='```json\n{"scores":[{"case_id":"case-1","score":1,'
            '"rationale":"Some context is missing.","confidence":"medium"}]}\n```'
        )

        result = _parse_scores(response, {"case-1"})

        self.assertEqual(result["case-1"]["score"], 1)

    def test_parses_root_array_response(self) -> None:
        response = SimpleNamespace(
            text='[{"case_id":"case-1","score":2,'
            '"rationale":"The idea is complete.","confidence":"high"}]'
        )

        result = _parse_scores(response, {"case-1"})

        self.assertEqual(result["case-1"]["score"], 2)

    def test_rejects_missing_or_invalid_score_rows(self) -> None:
        response = SimpleNamespace(
            text='{"scores":[{"case_id":"case-1","score":3,'
            '"rationale":"invalid score","confidence":"high"}]}'
        )

        with self.assertRaisesRegex(RuntimeError, "invalid score row"):
            _parse_scores(response, {"case-1"})

    def test_rejects_incomplete_batch_response(self) -> None:
        response = SimpleNamespace(
            text='{"scores":[{"case_id":"case-1","score":1,'
            '"rationale":"partial","confidence":"medium"}]}'
        )

        with self.assertRaisesRegex(RuntimeError, "case IDs mismatch"):
            _parse_scores(response, {"case-1", "case-2"})


if __name__ == "__main__":
    unittest.main()
