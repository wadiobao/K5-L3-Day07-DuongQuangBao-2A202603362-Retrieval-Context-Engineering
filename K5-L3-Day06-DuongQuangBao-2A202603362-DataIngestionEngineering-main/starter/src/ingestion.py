"""Load, normalize, and keep source locations for supported documents."""
import logging
import re
import unicodedata
from pathlib import Path
from urllib.request import Request, urlopen

from .models import DocumentPage

logger = logging.getLogger(__name__)

_GUTENBERG_START = "*** START OF THE PROJECT GUTENBERG EBOOK"
_GUTENBERG_END = "*** END OF THE PROJECT GUTENBERG EBOOK"
_CHAPTER = re.compile(
    r"^CHAPTER ([IVXLCDM]+)\.\s*\n([^\n]+)\n",
    re.MULTILINE,
)


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace while preserving paragraph boundaries."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00ad", "")
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    paragraphs = []
    for paragraph in re.split(r"\n\s*\n", text):
        normalized = re.sub(r"\s+", " ", paragraph).strip()
        if normalized:
            paragraphs.append(normalized)
    return "\n\n".join(paragraphs)


def load_pdf(path: Path, document_id: str, source: str) -> list[DocumentPage]:
    """Extract each physical PDF page and record pages with no text."""
    from pypdf import PdfReader

    pages: list[DocumentPage] = []
    for page_number, pdf_page in enumerate(PdfReader(str(path)).pages, start=1):
        text = normalize_text(pdf_page.extract_text() or "")
        if not text:
            logger.warning("PDF page has no extractable text: %s page %d", path, page_number)
        pages.append(
            DocumentPage(
                document_id=document_id,
                source=source,
                page=page_number,
                text=text,
                metadata={"document_id": document_id, "source": source, "page": page_number},
            )
        )
    return pages


def download_gutenberg_text(url: str, destination: Path) -> Path:
    """Download and validate a Project Gutenberg plain-text ebook."""
    request = Request(url, headers={"User-Agent": "Day06DataIngestionLab/1.0"})
    with urlopen(request, timeout=30) as response:
        content = response.read().decode("utf-8-sig")

    if _GUTENBERG_START not in content or _GUTENBERG_END not in content:
        raise ValueError(f"Downloaded text is not a complete Gutenberg ebook: {url}")

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_name(destination.name + ".tmp")
    temporary_path.write_text(content, encoding="utf-8")
    temporary_path.replace(destination)
    return destination


def load_gutenberg_text(
    path: Path,
    document_id: str,
    source: str,
    target_unit_chars: int = 3000,
    minimum_unit_chars: int = 1500,
) -> list[DocumentPage]:
    """Split a Gutenberg book into traceable, paragraph-aligned source units."""
    if target_unit_chars < minimum_unit_chars:
        raise ValueError("target_unit_chars must be at least minimum_unit_chars")
    if minimum_unit_chars <= 0:
        raise ValueError("minimum_unit_chars must be greater than zero")

    content = Path(path).read_text(encoding="utf-8-sig")
    try:
        content = content.split(_GUTENBERG_START, 1)[1].split(_GUTENBERG_END, 1)[0]
    except IndexError as exc:
        raise ValueError(f"Missing Project Gutenberg start/end markers in {path}") from exc

    chapter_matches = list(_CHAPTER.finditer(content))
    if not chapter_matches:
        raise ValueError(f"No chapter headings found in Project Gutenberg text: {path}")

    result: list[DocumentPage] = []
    for chapter_index, chapter_match in enumerate(chapter_matches):
        next_start = (
            chapter_matches[chapter_index + 1].start()
            if chapter_index + 1 < len(chapter_matches)
            else len(content)
        )
        chapter_text = normalize_text(content[chapter_match.end() : next_start])
        if not chapter_text:
            logger.warning("Empty chapter in source: %s", chapter_match.group(1))
            continue

        section_id = f"chapter-{chapter_match.group(1).lower()}"
        section = f"Chapter {chapter_match.group(1)} — {chapter_match.group(2).strip()}"
        paragraphs = list(
            re.finditer(r"\S.*?(?=\n\s*\n|\Z)", chapter_text, flags=re.DOTALL)
        )
        pieces: list[tuple[int, int]] = []
        for paragraph_match in paragraphs:
            paragraph_start = paragraph_match.start()
            paragraph_end = paragraph_match.end()
            paragraph = chapter_text[paragraph_start:paragraph_end].strip()
            if len(paragraph) <= target_unit_chars:
                pieces.append((paragraph_start, paragraph_end))
                continue

            word_matches = list(re.finditer(r"\S+", paragraph))
            part_start = word_matches[0].start()
            part_end = part_start
            for word_match in word_matches:
                if word_match.end() - part_start > target_unit_chars and part_end > part_start:
                    pieces.append(
                        (paragraph_start + part_start, paragraph_start + part_end)
                    )
                    part_start = word_match.start()
                part_end = word_match.end()
            if part_end > part_start:
                pieces.append((paragraph_start + part_start, paragraph_start + part_end))

        grouped: list[tuple[int, int]] = []
        group_start: int | None = None
        group_end: int | None = None
        for piece_start, piece_end in pieces:
            if (
                group_start is not None
                and group_end is not None
                and piece_end - group_start > target_unit_chars
            ):
                grouped.append((group_start, group_end))
                group_start = None
                group_end = None
            if group_start is None:
                group_start = piece_start
            group_end = piece_end
        if group_start is not None and group_end is not None:
            grouped.append((group_start, group_end))

        if len(grouped) > 1:
            last_start, last_end = grouped[-1]
            if last_end - last_start < minimum_unit_chars:
                previous_start, _ = grouped[-2]
                grouped[-2:] = [(previous_start, last_end)]

        for unit_start, unit_end in grouped:
            unit_text = chapter_text[unit_start:unit_end].strip()
            if len(unit_text) < minimum_unit_chars:
                logger.warning(
                    "Skipping short source unit in %s (%d characters)",
                    section_id,
                    len(unit_text),
                )
                continue
            page_number = len(result) + 1
            unit_id = f"{document_id}-unit-{page_number:03d}"
            result.append(
                DocumentPage(
                    document_id=document_id,
                    source=source,
                    page=page_number,
                    text=unit_text,
                    metadata={
                        "document_id": document_id,
                        "source": source,
                        "page": page_number,
                        "unit_id": unit_id,
                        "unit_start_char": unit_start,
                        "unit_end_char": unit_end,
                        "section_id": section_id,
                        "section": section,
                        "char_count": len(unit_text),
                    },
                )
            )

    return result
