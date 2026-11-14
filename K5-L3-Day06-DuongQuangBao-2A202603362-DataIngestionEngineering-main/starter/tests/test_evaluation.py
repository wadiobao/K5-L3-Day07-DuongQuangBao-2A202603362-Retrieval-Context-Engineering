import unittest

from src.evaluation import evaluate_query, summarize


class EvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.question = {
            "id": "q1",
            "split": "dev",
            "question": "What happened?",
            "gold_source_id": "book",
            "gold_page": 2,
            "evidence_quote": "Alice found the key.",
            "filter": {"section_id": "chapter-i"},
        }

    def test_evaluates_only_matching_page_with_sufficient_evidence(self) -> None:
        hit = {
            "document_id": "book",
            "page": 2,
            "text": "Alice found the key.",
            "metadata": {"section_id": "chapter-i"},
        }
        trace = evaluate_query(self.question, [hit])

        self.assertTrue(trace["hit1"])
        self.assertTrue(trace["evidence_hit3"])
        self.assertTrue(trace["filter_compliant"])

    def test_summarize_reports_counts_and_rates_by_split(self) -> None:
        traces = [
            {"strategy": "A", "split": "dev", "hit1": True, "evidence_hit3": True},
            {"strategy": "A", "split": "dev", "hit1": False, "evidence_hit3": True},
        ]
        result = summarize(traces)[0]

        self.assertEqual(result["hit1_count"], 1)
        self.assertEqual(result["evidence_hit3_count"], 2)
        self.assertEqual(result["evidence_hit3_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
