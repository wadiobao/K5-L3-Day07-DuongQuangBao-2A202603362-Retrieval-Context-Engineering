import tempfile
import unittest
from pathlib import Path

from src.reporting import ReportWriter


class ReportWriterTests(unittest.TestCase):
    def test_updates_sections_incrementally_and_preserves_prior_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            template = root / "template.md"
            output = root / "results" / "report.md"
            template.write_text(
                "# Report\nSetup: <!-- AUTO:setup -->\n<!-- AUTO:results -->\n",
                encoding="utf-8",
            )
            writer = ReportWriter(template, output)

            writer.update({"setup": "Gemini, 768 dimensions"})
            writer.update({"results": "Not run"})

            report = output.read_text(encoding="utf-8")
            self.assertIn("Setup: Gemini, 768 dimensions", report)
            self.assertIn("Not run", report)
            self.assertNotIn("<!-- AUTO:", report)

    def test_rejects_unknown_sections(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            template = root / "template.md"
            template.write_text("<!-- AUTO:setup -->", encoding="utf-8")
            writer = ReportWriter(template, root / "report.md")

            with self.assertRaisesRegex(ValueError, "Unknown report sections"):
                writer.update({"unknown": "value"})


if __name__ == "__main__":
    unittest.main()
