"""Download and index the World Development Report 2024 PDF."""
import argparse
import hashlib
import json
import re
import urllib.request
from pathlib import Path
from typing import Any


SOURCE_URL = 'https://bit.ly/WDR2024_FullReport_ENG'
SOURCE_ID = 'world-bank-wdr-2024'
SOURCE_NAME = 'World Development Report 2024: The Middle-Income Trap'
DOI = '10.1596/978-1-4648-2078-6'
LICENSE = 'CC BY 3.0 IGO'
CHUNK_CHARS = 4200
OVERLAP_CHARS = 250


def _download(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        SOURCE_URL,
        headers={'User-Agent': 'Day7-Retrieval-Context-Engineering/1.0'},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        content = response.read()
    if not content.startswith(b'%PDF-'):
        raise RuntimeError('Nguồn tải về không phải tệp PDF')
    destination.write_bytes(content)


def _split_text(text: str, limit: int = CHUNK_CHARS) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r'\n\s*\n', text) if part.strip()]
    chunks: list[str] = []
    current = ''
    for paragraph in paragraphs:
        while len(paragraph) > limit:
            if current:
                chunks.append(current)
                current = ''
            boundary = paragraph.rfind(' ', 0, limit)
            boundary = boundary if boundary > limit // 2 else limit
            chunks.append(paragraph[:boundary].strip())
            paragraph = paragraph[max(0, boundary - OVERLAP_CHARS):].strip()
        candidate = f'{current}\n\n{paragraph}' if current else paragraph
        if len(candidate) > limit and current:
            chunks.append(current)
            current = paragraph
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _load_visual_notes(path: Path) -> dict[int, list[dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(f'Thiếu chú giải hình ảnh đã kiểm chứng: {path}')
    notes = json.loads(path.read_text(encoding='utf-8'))
    by_page: dict[int, list[dict[str, Any]]] = {}
    for note in notes:
        if not all(note.get(key) for key in ('pdf_page', 'printed_page', 'element', 'kind', 'region', 'description')):
            raise ValueError(f'Chú giải hình ảnh không hợp lệ: {note}')
        by_page.setdefault(int(note['pdf_page']), []).append(note)
    return by_page


def build_index(pdf_path: Path, notes_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        import pymupdf
    except ImportError as exc:
        raise RuntimeError('Cần PyMuPDF; cài starter/requirements.txt') from exc

    notes = _load_visual_notes(notes_path)
    document = pymupdf.open(pdf_path)
    chunks: list[dict[str, Any]] = []
    blank_pages = []
    image_pages = []
    vector_pages = []
    extracted_chars = 0
    text_pages = 0
    figure_pages = set()
    table_pages = set()
    figure_pattern = re.compile(r'\bfigure\s+(?:[A-Z]\.\d+|\d+)(?:\.\d+)*', re.I)
    table_pattern = re.compile(r'\btable\s+(?:[A-Z]\.\d+|\d+)(?:\.\d+)*', re.I)

    for page_index, page in enumerate(document):
        pdf_page = page_index + 1
        text = page.get_text('text').strip()
        if text:
            text_pages += 1
            extracted_chars += len(text)
            figure_pages.update([pdf_page] if figure_pattern.search(text) else [])
            table_pages.update([pdf_page] if table_pattern.search(text) else [])
        else:
            blank_pages.append(pdf_page)
        if page.get_images(full=True):
            image_pages.append(pdf_page)
        if page.get_drawings():
            vector_pages.append(pdf_page)

        page_notes = notes.get(pdf_page, [])
        page_chunks = _split_text(text)
        for index, content in enumerate(page_chunks, start=1):
            chunks.append({
                'id': f'wdr2024-pdf{pdf_page:03d}-c{index:02d}',
                'source_id': SOURCE_ID,
                'source': SOURCE_NAME,
                'page': pdf_page,
                'text': content,
                'metadata': {
                    'pdf_page': pdf_page,
                    'visual_elements': [],
                    'contains_extracted_text': bool(text),
                },
            })

        for index, note in enumerate(page_notes, start=len(page_chunks) + 1):
            note_text = (
                f"VISUAL EVIDENCE — {note['element']} (printed page {note['printed_page']}; "
                f"{note['kind']}; region: {note['region']}): {note['description']}"
            )
            chunks.append({
                'id': f'wdr2024-pdf{pdf_page:03d}-c{index:02d}',
                'source_id': SOURCE_ID,
                'source': SOURCE_NAME,
                'page': pdf_page,
                'text': note_text,
                'metadata': {
                    'pdf_page': pdf_page,
                    'visual_elements': [{
                        'element': note['element'],
                        'kind': note['kind'],
                        'printed_page': note['printed_page'],
                        'region': note['region'],
                    }],
                    'contains_extracted_text': False,
                },
            })

    raw_pdf = pdf_path.read_bytes()
    manifest = {
        'document': SOURCE_NAME,
        'author': 'World Bank',
        'publication_year': 2024,
        'doi': DOI,
        'source_url': SOURCE_URL,
        'license': LICENSE,
        'attribution': f'World Bank. 2024. {SOURCE_NAME}. Washington, DC: World Bank. doi:{DOI}. License: {LICENSE}.',
        'sha256': hashlib.sha256(raw_pdf).hexdigest(),
        'pdf_bytes': len(raw_pdf),
        'pdf_pages': len(document),
        'pages_with_extracted_text': text_pages,
        'pages_with_no_extracted_text': blank_pages,
        'pages_with_embedded_raster_images': image_pages,
        'pages_with_vector_drawings': vector_pages,
        'pages_with_figure_labels': sorted(figure_pages),
        'pages_with_table_labels': sorted(table_pages),
        'manually_verified_visual_notes': sum(len(items) for items in notes.values()),
        'visual_note_pdf_pages': sorted(notes),
        'extracted_text_characters': extracted_chars,
        'chunk_count': len(chunks),
        'chunk_char_limit': CHUNK_CHARS,
        'overlap_chars_for_long_paragraphs': OVERLAP_CHARS,
        'extraction_method': 'PyMuPDF native PDF text/layout extraction; manually checked visual descriptions for figures and tables listed in visual_notes.json.',
        'unprocessed_content': (
            'No OCR was run. All pages with no selectable text are listed and were visually checked as blank; '
            'they are not indexed. Raster images are inventoried by page. Informative charts, diagrams, maps, '
            'and tables listed in visual_notes.json have manually checked text descriptions; the publication '
            'page image contains no additional evidence beyond its extracted text and decorative marks.'
        ),
    }
    return chunks, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description='Tải và lập chỉ mục báo cáo WDR 2024')
    parser.add_argument('--pdf', type=Path, default=Path('starter/advanced/source/wdr2024.pdf'))
    parser.add_argument('--output', type=Path, default=Path('starter/advanced/data'))
    args = parser.parse_args()
    if not args.pdf.exists():
        _download(args.pdf)
    notes_path = Path(__file__).with_name('visual_notes.json')
    chunks, manifest = build_index(args.pdf, notes_path)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'chunks.json').write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    (args.output / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    print(json.dumps({
        'pages': manifest['pdf_pages'],
        'text_pages': manifest['pages_with_extracted_text'],
        'chunks': manifest['chunk_count'],
        'manual_visual_notes': manifest['manually_verified_visual_notes'],
        'output': str(args.output),
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
