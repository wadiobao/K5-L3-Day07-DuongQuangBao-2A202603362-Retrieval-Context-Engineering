"""Chunk source units while retaining stable IDs and source locations."""
import hashlib
import re

from .models import Chunk, DocumentPage


def _chunk_id(page: DocumentPage, strategy: str, start: int, end: int) -> str:
    value = f"{page.document_id}|{page.page}|{strategy}|{start}|{end}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def _make_chunk(
    page: DocumentPage,
    strategy: str,
    text: str,
    start: int,
    end: int,
) -> Chunk:
    metadata = dict(page.metadata)
    metadata.update(
        {
            "document_id": page.document_id,
            "source": page.source,
            "page": page.page,
            "start_char": start,
            "end_char": end,
        }
    )
    return Chunk(
        chunk_id=_chunk_id(page, strategy, start, end),
        document_id=page.document_id,
        source=page.source,
        page=page.page,
        text=text,
        metadata=metadata,
    )


def fixed_size(page: DocumentPage, size: int, overlap: int) -> list[Chunk]:
    """Split text into fixed windows, retaining source-page and character offsets."""
    if size <= 0:
        raise ValueError("size must be greater than zero")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be non-negative and smaller than size")
    if not page.text:
        return []

    chunks: list[Chunk] = []
    start = 0
    while start < len(page.text):
        end = min(start + size, len(page.text))
        value = page.text[start:end].strip()
        if value:
            chunks.append(_make_chunk(page, "fixed", value, start, end))
        if end == len(page.text):
            break
        start = end - overlap
    return chunks


def _split_long_paragraph(
    paragraph: str,
    paragraph_start: int,
    max_size: int,
) -> list[tuple[int, int]]:
    words = list(re.finditer(r"\S+", paragraph))
    if not words:
        return []

    pieces: list[tuple[int, int]] = []
    piece_start = words[0].start()
    piece_end = piece_start
    for word in words:
        if word.end() - piece_start > max_size and piece_end > piece_start:
            pieces.append(
                (paragraph_start + piece_start, paragraph_start + piece_end)
            )
            piece_start = word.start()
        piece_end = word.end()
    if piece_end > piece_start:
        pieces.append((paragraph_start + piece_start, paragraph_start + piece_end))
    return pieces


def by_paragraph(page: DocumentPage, max_size: int) -> list[Chunk]:
    """Group whole paragraphs up to a size limit and split oversized paragraphs."""
    if max_size <= 0:
        raise ValueError("max_size must be greater than zero")
    if not page.text:
        return []

    paragraphs = list(
        re.finditer(r"\S.*?(?=\n\s*\n|\Z)", page.text, flags=re.DOTALL)
    )
    pieces: list[tuple[int, int]] = []
    for match in paragraphs:
        start, end = match.start(), match.end()
        if end - start <= max_size:
            pieces.append((start, end))
        else:
            pieces.extend(
                _split_long_paragraph(page.text[start:end], start, max_size)
            )

    chunks: list[Chunk] = []
    group_start: int | None = None
    group_end: int | None = None
    for piece_start, piece_end in pieces:
        if (
            group_start is not None
            and group_end is not None
            and piece_end - group_start > max_size
        ):
            value = page.text[group_start:group_end].strip()
            if value:
                chunks.append(
                    _make_chunk(page, "paragraph", value, group_start, group_end)
                )
            group_start = None
            group_end = None
        if group_start is None:
            group_start = piece_start
        group_end = piece_end

    if group_start is not None and group_end is not None:
        value = page.text[group_start:group_end].strip()
        if value:
            chunks.append(
                _make_chunk(page, "paragraph", value, group_start, group_end)
            )
    return chunks
