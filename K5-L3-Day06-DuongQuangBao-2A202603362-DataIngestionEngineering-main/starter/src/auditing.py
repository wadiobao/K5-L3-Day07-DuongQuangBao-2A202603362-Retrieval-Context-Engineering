"""Generate deterministic prechecks without filling human evaluation labels."""
import csv
import hashlib
from pathlib import Path
from typing import Any

from .ingestion import normalize_text
from .models import DocumentPage

_METADATA_FIELDS = [
    "question_id",
    "strategy",
    "rank",
    "chunk_id",
    "document_id",
    "source",
    "expected_source",
    "page",
    "page_exists",
    "section_id",
    "expected_section_id",
    "required_metadata_present",
    "metadata_matches_source_page",
    "filter_compliant",
    "metadata_useful",
    "rationale",
]
_GROUNDING_FIELDS = [
    "question_id",
    "strategy",
    "question",
    "evidence_hit3",
    "exact_evidence_ranks",
    "automatic_evidence_status",
    "human_score_0_to_2",
    "human_rationale",
]
_UNIT_FIELDS = [
    "unit_id",
    "document_id",
    "logical_page",
    "source",
    "section_id",
    "section",
    "source_start_char",
    "source_end_char",
    "char_count",
    "text_sha256",
]


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_source_unit_inventory(
    output_dir: Path,
    pages: list[DocumentPage],
) -> Path:
    """Record the generated logical-page boundaries and normalized lengths."""
    rows = [
        {
            "unit_id": page.metadata["unit_id"],
            "document_id": page.document_id,
            "logical_page": page.page,
            "source": page.source,
            "section_id": page.metadata["section_id"],
            "section": page.metadata["section"],
            "source_start_char": page.metadata["unit_start_char"],
            "source_end_char": page.metadata["unit_end_char"],
            "char_count": len(page.text),
            "text_sha256": hashlib.sha256(page.text.encode("utf-8")).hexdigest(),
        }
        for page in pages
    ]
    return _write_csv(
        Path(output_dir) / "source_unit_inventory.csv",
        rows,
        _UNIT_FIELDS,
    )


def write_metadata_precheck(
    output_dir: Path,
    pages: list[DocumentPage],
    traces: list[dict[str, Any]],
) -> Path:
    """Compare retrieved metadata with the source-unit inventory."""
    page_index = {(page.document_id, page.page): page for page in pages}
    rows: list[dict[str, Any]] = []
    for trace in traces:
        if trace["split"] != "test":
            continue
        filters = trace["filters"]
        for rank, hit in enumerate(trace["hits"], start=1):
            document_id = str(hit.get("document_id", ""))
            try:
                page_number = int(hit["page"])
            except (KeyError, TypeError, ValueError):
                page_number = -1
            page = page_index.get((document_id, page_number))
            metadata = hit.get("metadata") or {}
            expected_source = page.source if page else ""
            expected_section = page.metadata["section_id"] if page else ""
            required_present = all(
                field in metadata
                for field in ("document_id", "source", "page", "section_id")
            )
            metadata_matches = bool(
                page
                and metadata.get("document_id") == page.document_id
                and metadata.get("source") == page.source
                and metadata.get("page") == page.page
                and metadata.get("section_id") == expected_section
                and hit.get("source") == page.source
                and hit.get("page") == page.page
            )
            filter_compliant: bool | str = "N/A"
            if filters:
                filter_compliant = all(
                    metadata.get(key) == value for key, value in filters.items()
                )
            rows.append(
                {
                    "question_id": trace["question_id"],
                    "strategy": trace["strategy"],
                    "rank": rank,
                    "chunk_id": hit.get("chunk_id", ""),
                    "document_id": document_id,
                    "source": hit.get("source", ""),
                    "expected_source": expected_source,
                    "page": hit.get("page", ""),
                    "page_exists": page is not None,
                    "section_id": metadata.get("section_id", ""),
                    "expected_section_id": expected_section,
                    "required_metadata_present": required_present,
                    "metadata_matches_source_page": metadata_matches,
                    "filter_compliant": filter_compliant,
                    "metadata_useful": "",
                    "rationale": "",
                }
            )
    return _write_csv(
        Path(output_dir) / "metadata_precheck.csv",
        rows,
        _METADATA_FIELDS,
    )


def write_grounding_precheck(
    output_dir: Path,
    traces: list[dict[str, Any]],
) -> Path:
    """Summarize exact gold-evidence matches per test question and strategy."""
    rows: list[dict[str, Any]] = []
    for trace in traces:
        if trace["split"] != "test":
            continue
        locations = {
            (item["gold_source_id"], int(item["gold_page"]))
            for item in trace["gold_locations"]
        }
        evidence = normalize_text(trace["evidence_quote"]).casefold()
        matched_ranks = []
        for rank, hit in enumerate(trace["hits"], start=1):
            location = (hit.get("document_id", ""), hit.get("page"))
            text = normalize_text(hit.get("text", "")).casefold()
            if location in locations and evidence in text:
                matched_ranks.append(rank)
        rows.append(
            {
                "question_id": trace["question_id"],
                "strategy": trace["strategy"],
                "question": trace["question"],
                "evidence_hit3": bool(matched_ranks),
                "exact_evidence_ranks": ",".join(map(str, matched_ranks)),
                "automatic_evidence_status": (
                    "exact gold evidence in top-3"
                    if matched_ranks
                    else "no exact gold evidence in top-3"
                ),
                "human_score_0_to_2": "",
                "human_rationale": "",
            }
        )
    return _write_csv(
        Path(output_dir) / "grounding_precheck.csv",
        rows,
        _GROUNDING_FIELDS,
    )


def write_manual_review_guide(
    output_dir: Path,
    coherence_count: int,
    metadata_count: int,
    grounding_count: int,
) -> Path:
    """Tell the student exactly which generated fields still require judgment."""
    content = f"""# Các bước chấm thủ công còn lại

Các precheck tự động chỉ là đối chiếu dữ liệu, không phải điểm rubric. Nếu có `coherence_ai_suggestions.csv` hoặc `grounding_ai_suggestions.csv`, đó chỉ là gợi ý; hãy xem trace và nguồn rồi tự điền nhãn cuối.

1. **Coherence — {coherence_count} dòng:** trong `coherence_audit.csv`, điền `score_0_to_2` và `rationale`. Chấm 0 nếu chunk bị cắt/không hiểu được, 1 nếu thiếu một phần ngữ cảnh, 2 nếu tự hiểu được và giữ trọn ý.
2. **Metadata — {metadata_count} hit:** trong `metadata_grounding_audit.csv`, xác nhận cột `metadata_correct`, đánh giá `metadata_useful`, ghi `rationale`; đối chiếu các cột `metadata_precheck.csv` với `source_unit_inventory.csv`. Với mọi cấu hình, kiểm tra top-3 có khớp filter nêu trong câu hỏi hay không; C là cấu hình thực sự áp dụng filter.
3. **Grounding — {grounding_count} cặp câu/strategy:** trong `grounding_precheck.csv`, chấm **top-3 gộp lại**: 0 = không có bằng chứng; 1 = có bằng chứng nhưng thiếu điều kiện/ngữ cảnh; 2 = đủ để trả lời và truy nguồn. Điền `human_score_0_to_2` và `human_rationale`. `exact_evidence_ranks` chỉ là gợi ý tự động, không tự quyết định điểm.
4. **Chi phí:** kiểm tra Google AI Studio, rồi cập nhật số tiền thực tế (hoặc 0 nếu xác nhận không bị tính phí) vào `run_metadata.json` và báo cáo. Không suy ra chi phí từ số request.
5. **Thông tin nộp bài:** hoàn tất các trường còn trống trong `../STUDENT.md` (chủ đề/corpus, repo bài nộp, commit chốt).

Sau khi chấm, tổng hợp nhãn và ví dụ lỗi vào báo cáo cá nhân. Không đổi nhãn thủ công thành kết quả tự động.
"""
    path = Path(output_dir) / "manual_review.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path
