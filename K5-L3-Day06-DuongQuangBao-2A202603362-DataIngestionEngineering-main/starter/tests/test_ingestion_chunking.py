import tempfile
import unittest
from pathlib import Path

from src.chunking import by_paragraph, fixed_size
from src.ingestion import load_gutenberg_text, normalize_text
from src.models import DocumentPage


class IngestionTests(unittest.TestCase):
    def test_normalize_text_preserves_paragraph_boundaries(self) -> None:
        self.assertEqual(
            normalize_text(" A  line\r\ncontinues \r\n\r\n  New paragraph\u00ad "),
            "A line continues\n\nNew paragraph",
        )

    def test_gutenberg_loader_assigns_stable_source_units(self) -> None:
        paragraph = "word " * 520
        book = (
            "header\n*** START OF THE PROJECT GUTENBERG EBOOK Sample ***\n"
            "CHAPTER I.\nFirst\n\n"
            f"{paragraph}\n\n{paragraph}\n\n"
            "CHAPTER II.\nSecond\n\n"
            f"{paragraph}\n\n{paragraph}\n"
            "*** END OF THE PROJECT GUTENBERG EBOOK Sample ***\n"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.txt"
            path.write_text(book, encoding="utf-8")
            pages = load_gutenberg_text(
                path,
                document_id="sample",
                source="sample.txt",
                target_unit_chars=1600,
                minimum_unit_chars=1500,
            )

        self.assertGreaterEqual(len(pages), 2)
        self.assertEqual(pages[0].metadata["section_id"], "chapter-i")
        self.assertEqual(pages[-1].metadata["section_id"], "chapter-ii")
        self.assertTrue(all(len(page.text) >= 1500 for page in pages))
        self.assertTrue(
            all(
                page.metadata["unit_start_char"] < page.metadata["unit_end_char"]
                for page in pages
            )
        )


class ChunkingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.page = DocumentPage(
            document_id="doc",
            source="source.txt",
            page=7,
            text="First paragraph with several words.\n\nSecond paragraph has more text.",
            metadata={"section_id": "chapter-i"},
        )

    def test_fixed_size_retains_location_and_overlap(self) -> None:
        chunks = fixed_size(self.page, size=32, overlap=8)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.page == 7 for chunk in chunks))
        self.assertTrue(all(chunk.metadata["section_id"] == "chapter-i" for chunk in chunks))
        self.assertNotEqual(chunks[0].chunk_id, chunks[1].chunk_id)

    def test_paragraph_chunking_keeps_paragraphs_together(self) -> None:
        chunks = by_paragraph(self.page, max_size=50)

        self.assertEqual(len(chunks), 2)
        self.assertTrue(all("\n\n" not in chunk.text for chunk in chunks))
        self.assertTrue(all(chunk.metadata["page"] == 7 for chunk in chunks))

    def test_rejects_invalid_chunk_settings(self) -> None:
        with self.assertRaises(ValueError):
            fixed_size(self.page, size=10, overlap=10)
        with self.assertRaises(ValueError):
            by_paragraph(self.page, max_size=0)


if __name__ == "__main__":
    unittest.main()
