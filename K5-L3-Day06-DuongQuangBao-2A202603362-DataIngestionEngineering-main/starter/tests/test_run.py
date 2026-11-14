import csv
import tempfile
import unittest
from pathlib import Path

from src.models import Chunk, DocumentPage
from src.run import _manual_table, _write_audits


class RunAuditTests(unittest.TestCase):
    def test_filter_compliance_is_scored_for_every_strategy(self) -> None:
        question = {
            "id": "q1",
            "split": "test",
            "question": "What happened?",
            "filter": {"section_id": "chapter-i"},
        }
        pages = [
            DocumentPage(
                "book",
                "source",
                1,
                "A sample source paragraph.",
                {"section_id": "chapter-i"},
            )
        ]
        chunks = {
            strategy: [
                Chunk(
                    f"{strategy}-chunk",
                    "book",
                    "source",
                    1,
                    "A sample source paragraph.",
                    {"section_id": "chapter-i"},
                )
            ]
            for strategy in ("A", "B", "C")
        }
        traces = [
            {
                "strategy": strategy,
                "question_id": "q1",
                "split": "test",
                "question": "What happened?",
                "filters": question["filter"],
                "evidence_quote": "A sample source paragraph.",
                "gold_locations": [
                    {"gold_source_id": "book", "gold_page": 1}
                ],
                "filter_compliant": strategy == "C",
                "hits": [
                    {
                        "chunk_id": f"{strategy}-chunk",
                        "text": "A sample source paragraph.",
                        "source": "source",
                        "page": 1,
                        "metadata": {
                            "section_id": (
                                "chapter-i" if strategy == "C" else "chapter-ii"
                            )
                        },
                    }
                ],
            }
            for strategy in ("A", "B", "C")
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            _, manual_path = _write_audits(
                Path(temporary_directory),
                pages,
                chunks,
                [question],
                traces,
            )
            with manual_path.open(encoding="utf-8", newline="") as file:
                audit_rows = list(csv.DictReader(file))

        compliance = {
            row["strategy"]: row["filter_compliant"] for row in audit_rows
        }
        self.assertEqual(compliance, {"A": "False", "B": "False", "C": "True"})

        report_rows = _manual_table(traces).splitlines()
        self.assertIn("| A | Chưa chấm /2 | Chưa chấm | 0/1 (0.0%) | Chưa chấm |", report_rows)
        self.assertIn("| B | Chưa chấm /2 | Chưa chấm | 0/1 (0.0%) | Chưa chấm |", report_rows)
        self.assertIn("| C | Chưa chấm /2 | Chưa chấm | 1/1 (100.0%) | Chưa chấm |", report_rows)


if __name__ == "__main__":
    unittest.main()
