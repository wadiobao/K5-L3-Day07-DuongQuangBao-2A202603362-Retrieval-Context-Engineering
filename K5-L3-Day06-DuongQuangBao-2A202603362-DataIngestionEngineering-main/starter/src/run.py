"""Prepare and run the Day 06 ingestion and retrieval benchmark."""
import argparse
import csv
import json
import logging
import os
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .auditing import (
    write_grounding_precheck,
    write_manual_review_guide,
    write_metadata_precheck,
    write_source_unit_inventory,
)
from .chunking import by_paragraph, fixed_size
from .embeddings import MyEmbedder
from .evaluation import evaluate_query, summarize
from .ingestion import download_gutenberg_text, load_gutenberg_text
from .models import Chunk, DocumentPage
from .reporting import ReportWriter
from .store import PgVectorStore

logger = logging.getLogger("day06")
STARTER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STARTER_ROOT.parent
DATA_ROOT = REPO_ROOT / "data"
BENCHMARK_ROOT = REPO_ROOT / "benchmark"
RESULTS_ROOT = REPO_ROOT / "results"
REPORT_PATH = REPO_ROOT / "report" / "individual" / "2A202603362_DuongQuangBao.md"

STRATEGY_ORDER = ("A", "B", "C")
FIXED_CHUNK_SIZE = 1500
FIXED_OVERLAP = 100
PARAGRAPH_CHUNK_SIZE = 1500


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _load_corpus(manifest: dict[str, Any]) -> list[DocumentPage]:
    path = DATA_ROOT / manifest["source_path"]
    if not path.exists():
        logger.info("Corpus is missing; downloading from the declared source URL")
        download_gutenberg_text(manifest["source_url"], path)

    pages = load_gutenberg_text(
        path,
        document_id=manifest["document_id"],
        source=manifest["source_url"],
        target_unit_chars=int(manifest["unit_target_chars"]),
        minimum_unit_chars=int(manifest["minimum_unit_chars"]),
    )
    for page in pages:
        page.metadata["language"] = manifest["language"]
        page.metadata["work_title"] = manifest["title"]

    minimum_chars = int(manifest["minimum_unit_chars"])
    valid_pages = [page for page in pages if len(page.text) >= minimum_chars]
    if len(valid_pages) < 30:
        raise ValueError(
            f"Corpus has {len(valid_pages)} source units of at least "
            f"{minimum_chars} characters; GUIDE.md requires at least 30"
        )
    expected_count = manifest.get("source_unit_count")
    if expected_count is not None and len(valid_pages) != int(expected_count):
        raise ValueError(
            f"Manifest expects {expected_count} source units; "
            f"loaded {len(valid_pages)}"
        )
    return valid_pages


def _load_benchmark(
    path: Path,
    pages: list[DocumentPage],
) -> list[dict[str, Any]]:
    questions = _read_json(path)
    if not isinstance(questions, list):
        raise ValueError(f"Benchmark must be a JSON array: {path}")
    if len(questions) < 20:
        raise ValueError(f"Benchmark has {len(questions)} questions; at least 20 are required")

    page_by_key = {(page.document_id, page.page): page for page in pages}
    known_ids: set[str] = set()
    resolved: list[dict[str, Any]] = []
    for raw_question in questions:
        question = dict(raw_question)
        question_id = question.get("id")
        if not question_id or question_id in known_ids:
            raise ValueError(f"Missing or duplicate benchmark question ID: {question_id!r}")
        known_ids.add(question_id)
        if question.get("split") not in {"dev", "test"}:
            raise ValueError(f"Question {question_id} split must be dev or test")
        for field in ("question", "expected_answer", "evidence_quote", "gold_source_id"):
            if not question.get(field):
                raise ValueError(f"Question {question_id} is missing {field}")
        try:
            page_number = int(question["gold_page"])
            gold_page = page_by_key[(question["gold_source_id"], page_number)]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Question {question_id} points to an unknown gold page") from exc

        evidence = question["evidence_quote"].casefold()
        if evidence not in gold_page.text.casefold():
            raise ValueError(
                f"Evidence for {question_id} is not present on gold page "
                f"{question['gold_source_id']}:{page_number}"
            )
        question["gold_page"] = page_number
        question["gold_section_id"] = gold_page.metadata["section_id"]

        filters = question.get("filter", question.get("filters")) or {}
        if not isinstance(filters, dict):
            raise ValueError(f"Question {question_id} filter must be a JSON object")
        question["filter"] = filters
        if filters and not any(
            all(page.metadata.get(key) == value for key, value in filters.items())
            for page in pages
        ):
            raise ValueError(f"Question {question_id} filter values do not exist in corpus")
        resolved.append(question)

    dev_count = sum(question["split"] == "dev" for question in resolved)
    test_count = sum(question["split"] == "test" for question in resolved)
    filtered = [question for question in resolved if question["filter"]]
    filtered_test = [question for question in filtered if question["split"] == "test"]
    if dev_count < 15 or test_count < 5:
        raise ValueError(f"Benchmark needs at least 15 dev / 5 test, got {dev_count}/{test_count}")
    if len(filtered) < 3 or not filtered_test:
        raise ValueError("Benchmark needs at least 3 filter questions, including one test question")
    return resolved


def _build_strategies(pages: list[DocumentPage]) -> dict[str, list[Chunk]]:
    fixed_chunks = [
        chunk
        for page in pages
        for chunk in fixed_size(page, FIXED_CHUNK_SIZE, FIXED_OVERLAP)
    ]
    paragraph_chunks = [
        chunk
        for page in pages
        for chunk in by_paragraph(page, PARAGRAPH_CHUNK_SIZE)
    ]
    return {"A": fixed_chunks, "B": paragraph_chunks, "C": paragraph_chunks}


def _chunk_stats(chunks_by_strategy: dict[str, list[Chunk]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for strategy in STRATEGY_ORDER:
        chunks = chunks_by_strategy[strategy]
        lengths = [len(chunk.text) for chunk in chunks]
        result[strategy] = {
            "chunk_count": len(chunks),
            "total_characters": sum(lengths),
            "min_characters": min(lengths) if lengths else 0,
            "median_characters": statistics.median(lengths) if lengths else 0,
            "max_characters": max(lengths) if lengths else 0,
        }
    return result


def _metric_text(count: int, total: int) -> str:
    return f"{count}/{total} ({count / total:.1%})" if total else "0/0 (N/A)"


def _retrieval_table(
    summary: list[dict[str, Any]],
    chunk_stats: dict[str, Any],
    elapsed_by_strategy: dict[str, float] | None = None,
) -> str:
    rows = []
    for item in summary:
        strategy = item["strategy"]
        elapsed = (elapsed_by_strategy or {}).get(strategy)
        elapsed_text = f"{elapsed:.2f}s" if elapsed is not None else "—"
        if elapsed is not None and strategy == "C":
            elapsed_text += " (incremental)"
        rows.append(
            "| {strategy} | {split} | {hit1} | {hit3} | {chunks} | {elapsed} | "
            "Check Gemini billing |".format(
                strategy=strategy,
                split=item["split"],
                hit1=_metric_text(item["hit1_count"], item["question_count"]),
                hit3=_metric_text(
                    item["evidence_hit3_count"], item["question_count"]
                ),
                chunks=chunk_stats[strategy]["chunk_count"],
                elapsed=elapsed_text,
            )
        )
    return "\n".join(rows) if rows else "| Chưa chạy | — | — | — | — | — | — |"


def _manual_table(
    traces: list[dict[str, Any]],
) -> str:
    rows = []
    for strategy in STRATEGY_ORDER:
        strategy_traces = [
            trace for trace in traces if trace["strategy"] == strategy
        ]
        filtered_traces = [
            trace
            for trace in strategy_traces
            if trace["filters"]
        ]
        compliant = sum(bool(trace["filter_compliant"]) for trace in filtered_traces)
        checked = len(filtered_traces)
        filter_text = _metric_text(compliant, checked) if checked else "N/A"
        rows.append(
            f"| {strategy} | Chưa chấm /2 | Chưa chấm | {filter_text} | Chưa chấm |"
        )
    return "\n".join(rows)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_audits(
    output_dir: Path,
    pages: list[DocumentPage],
    chunks_by_strategy: dict[str, list[Chunk]],
    test_questions: list[dict[str, Any]],
    traces: list[dict[str, Any]],
) -> tuple[Path, Path]:
    coherence_path = output_dir / "coherence_audit.csv"
    coherence_rows = []
    for page in pages[:10]:
        for strategy in STRATEGY_ORDER:
            matching = [
                chunk
                for chunk in chunks_by_strategy[strategy]
                if chunk.document_id == page.document_id and chunk.page == page.page
            ]
            if matching:
                chunk = matching[0]
                coherence_rows.append(
                    {
                        "strategy": strategy,
                        "document_id": page.document_id,
                        "source": page.source,
                        "page": page.page,
                        "section_id": page.metadata["section_id"],
                        "chunk_id": chunk.chunk_id,
                        "chunk_text": chunk.text,
                        "score_0_to_2": "",
                        "rationale": "",
                    }
                )
    _write_csv(coherence_path, coherence_rows)

    manual_path = output_dir / "metadata_grounding_audit.csv"
    trace_by_key = {(trace["strategy"], trace["question_id"]): trace for trace in traces}
    manual_rows = []
    for question in test_questions[:5]:
        for strategy in STRATEGY_ORDER:
            trace = trace_by_key[(strategy, question["id"])]
            for rank, hit in enumerate(trace["hits"], start=1):
                metadata = hit["metadata"]
                filters = question["filter"]
                filter_ok = all(metadata.get(key) == value for key, value in filters.items())
                manual_rows.append(
                    {
                        "question_id": question["id"],
                        "strategy": strategy,
                        "rank": rank,
                        "question": question["question"],
                        "chunk_id": hit["chunk_id"],
                        "text": hit["text"],
                        "source": hit["source"],
                        "page": hit["page"],
                        "section_id": metadata.get("section_id", ""),
                        "metadata_correct": "",
                        "metadata_useful": "",
                        "filter_compliant": filter_ok if filters else "N/A",
                        "rationale": "",
                    }
                )
    _write_csv(manual_path, manual_rows)
    write_metadata_precheck(output_dir, pages, traces)
    write_grounding_precheck(output_dir, traces)
    write_manual_review_guide(
        output_dir,
        len(coherence_rows),
        len(manual_rows),
        sum(trace["split"] == "test" for trace in traces),
    )
    return coherence_path, manual_path


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)


def _write_experiment_log(
    model_name: str,
    chunk_stats: dict[str, Any],
    summary: list[dict[str, Any]] | None = None,
) -> Path:
    summary_by_key = {
        (item["strategy"], item["split"]): item for item in (summary or [])
    }
    configurations = {
        "A": {
            "chunking": "fixed_size",
            "chunk_size": str(FIXED_CHUNK_SIZE),
            "overlap": str(FIXED_OVERLAP),
            "metadata": "document_id,source,page,section_id",
            "filter": "none",
            "hypothesis": "Fixed windows provide a reproducible baseline; may split paragraph context.",
        },
        "B": {
            "chunking": "paragraph",
            "chunk_size": str(PARAGRAPH_CHUNK_SIZE),
            "overlap": "0",
            "metadata": "document_id,source,page,section_id",
            "filter": "none",
            "hypothesis": "Paragraph-aware chunks may preserve coherent evidence better than fixed windows.",
        },
        "C": {
            "chunking": "paragraph",
            "chunk_size": str(PARAGRAPH_CHUNK_SIZE),
            "overlap": "0",
            "metadata": "document_id,source,page,section_id",
            "filter": "section_id when supplied by question",
            "hypothesis": "Filtering by a question-provided chapter may reduce cross-section noise.",
        },
    }
    fields = [
        "strategy",
        "chunking",
        "chunk_size",
        "overlap",
        "metadata",
        "filter",
        "vectorizer",
        "hypothesis",
        "dev_hit1",
        "dev_hit3",
        "test_hit1",
        "test_hit3",
        "notes",
    ]
    path = REPO_ROOT / "artifacts" / "experiment_log.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for strategy in STRATEGY_ORDER:
            config = configurations[strategy]
            row: dict[str, str] = {
                **config,
                "strategy": strategy,
                "vectorizer": model_name,
                "notes": (
                    f"{chunk_stats[strategy]['chunk_count']} chunks; "
                    "C reuses B document embeddings to isolate filter impact."
                ),
            }
            for split, result_field in (("dev", "dev"), ("test", "test")):
                metrics = summary_by_key.get((strategy, split))
                if metrics:
                    row[f"{result_field}_hit1"] = _metric_text(
                        metrics["hit1_count"], metrics["question_count"]
                    )
                    row[f"{result_field}_hit3"] = _metric_text(
                        metrics["evidence_hit3_count"], metrics["question_count"]
                    )
                else:
                    row[f"{result_field}_hit1"] = ""
                    row[f"{result_field}_hit3"] = ""
            writer.writerow(row)
    return path


def _dev_candidate(summary: list[dict[str, Any]]) -> str:
    dev_rows = [item for item in summary if item["split"] == "dev"]
    if not dev_rows:
        return "Chưa có kết quả dev; chưa thể chọn."
    ranked = sorted(
        dev_rows,
        key=lambda item: (
            item["evidence_hit3_rate"],
            item["hit1_rate"],
            -item["question_count"],
        ),
        reverse=True,
    )
    best = ranked[0]
    return (
        f"Ứng viên tạm thời theo dev: {best['strategy']} "
        f"(Evidence Hit@3 {best['evidence_hit3_count']}/{best['question_count']}); "
        "cần đọc nhãn grounding/coherence trước khi kết luận."
    )


def _write_report(
    writer: ReportWriter,
    manifest: dict[str, Any],
    pages: list[DocumentPage],
    questions: list[dict[str, Any]],
    chunk_stats: dict[str, Any],
    model_name: str,
    dimensions: int,
    traces: list[dict[str, Any]] | None = None,
    summary: list[dict[str, Any]] | None = None,
    elapsed_by_strategy: dict[str, float] | None = None,
    database: dict[str, str] | None = None,
    request_count: int | None = None,
    input_tokens: int | None = None,
    audit_paths: tuple[Path, Path] | None = None,
    query_embedding_seconds: float | None = None,
) -> None:
    traces = traces or []
    summary = summary or []
    filters = [question for question in questions if question["filter"]]
    char_count = sum(len(page.text) for page in pages)
    if summary:
        dev_metrics = {
            (item["strategy"], item["split"]): item
            for item in summary
        }
        a = dev_metrics.get(("A", "dev"))
        b = dev_metrics.get(("B", "dev"))
        c = dev_metrics.get(("C", "dev"))
        comparison = (
            f"Dev Evidence Hit@3 A/B/C: "
            f"{a['evidence_hit3_count']}/{a['question_count'] if a else 0}, "
            f"{b['evidence_hit3_count']}/{b['question_count'] if b else 0}, "
            f"{c['evidence_hit3_count']}/{c['question_count'] if c else 0}."
            if a and b and c
            else "Chưa đủ kết quả dev để so sánh A/B/C."
        )
        misses_by_question: dict[str, dict[str, Any]] = {}
        for trace in traces:
            if trace["split"] == "test" and not trace["evidence_hit3"]:
                miss = misses_by_question.setdefault(
                    trace["question_id"],
                    {
                        "gold": trace["gold_locations"][0],
                        "strategies": [],
                    },
                )
                miss["strategies"].append(trace["strategy"])
        error_examples = "; ".join(
            f"{question_id} (gold {miss['gold']['gold_source_id']}:"
            f"{miss['gold']['gold_page']}; miss ở "
            f"{'/'.join(miss['strategies'])})"
            for question_id, miss in list(misses_by_question.items())[:2]
        ) or "Không có test miss Evidence Hit@3."
        if len(misses_by_question) == 1:
            error_examples = f"Chỉ có 1 câu test miss độc lập: {error_examples}"
        analysis = (
            f"{comparison} A–B đo chunking; B–C đo filter. "
            f"Test miss examples: {error_examples}. Test chỉ báo cáo, "
            "không dùng để chọn cấu hình."
        )
        next_steps = (
            f"Chấm coherence/metadata/grounding trong "
            f"{audit_paths[0].name if audit_paths else 'coherence_audit.csv'} và "
            f"{audit_paths[1].name if audit_paths else 'metadata_grounding_audit.csv'}; "
            "xem `results/manual_review.md` để biết trường cần điền."
        )
    else:
        analysis = "Chưa chạy truy xuất; chưa có phép so sánh hay lỗi thực nghiệm."
        next_steps = "Chạy `python -m src.run` sau khi xác nhận cấu hình API và database."

    manual_results = (
        _manual_table(traces)
        if traces
        else "\n".join(
            f"| {strategy} | Chưa chấm /2 | Chưa chấm | N/A | Chưa chấm |"
            for strategy in STRATEGY_ORDER
        )
    )
    table = _retrieval_table(summary, chunk_stats, elapsed_by_strategy)
    permission = manifest["license_or_permission"]
    if database:
        store_text = (
            f"Gemini Embeddings API; `{model_name}`, {dimensions} chiều. "
            f"PostgreSQL {database['postgres_version']} + pgvector "
            f"{database['pgvector_version']}; cosine distance, exact search."
        )
    else:
        store_text = (
            f"Gemini Embeddings API; `{model_name}`, {dimensions} chiều (dự kiến). "
            "PostgreSQL + pgvector; database chưa được kiểm tra."
        )
    if request_count is not None:
        store_text += (
            f" Số request embedding: {request_count}; "
            f"token theo metadata API: {input_tokens if input_tokens is not None else 'không trả về'}. "
            "Chi phí cần xác minh trong Google AI Studio."
        )
    if query_embedding_seconds is not None:
        store_text += f" Query embedding chung: {query_embedding_seconds:.2f}s."

    writer.update(
        {
            "corpus_pipeline": (
                f"{manifest['title']} — {manifest['author']}; nguồn {manifest['source_url']}; "
                f"quyền: {permission}; "
                f"{len(pages)} đơn vị nội dung từ ít nhất {manifest['minimum_unit_chars']} "
                f"ký tự/đơn vị ({char_count:,} ký tự đã chuẩn hóa). "
                f"Page là logical source unit (không phải trang in): "
                f"{manifest['page_numbering_rule']} "
                f"Quy tắc chia: {manifest['unitization_rule']}"
            ),
            "normalization": (
                f"Project Gutenberg header/footer được bỏ; NFC/whitespace chuẩn hóa; "
                f"{len(pages)} đơn vị hợp lệ, lỗi/rỗng được log."
            ),
            "model_store": store_text,
            "metadata": (
                "Mỗi chunk có `document_id`, `source`, `page`, `section_id`, `section`, "
                "`language`; filter `section_id` lấy từ tiêu đề chương trong nguồn."
            ),
            "strategies": (
                f"A fixed {FIXED_CHUNK_SIZE} chars / overlap {FIXED_OVERLAP}; "
                f"B paragraph-aware, tối đa {PARAGRAPH_CHUNK_SIZE} chars; "
                "C giống B và chỉ thêm filter theo section khi câu hỏi nêu chương. "
                "Thời gian A/B gồm embedding riêng; C là thời gian tăng thêm vì tái dùng "
                "embedding của B. Thời gian embedding câu hỏi dùng chung không được cộng "
                "vào từng strategy. "
                f"Chunk counts: A={chunk_stats['A']['chunk_count']}, "
                f"B={chunk_stats['B']['chunk_count']}, C={chunk_stats['C']['chunk_count']}."
            ),
            "benchmark": (
                f"{len(questions)} câu, "
                f"{sum(q['split'] == 'dev' for q in questions)} dev / "
                f"{sum(q['split'] == 'test' for q in questions)} test; "
                f"{len(filters)} câu filter, "
                f"{sum(q['split'] == 'test' for q in filters)} thuộc test. "
                "Evidence được kiểm tra khớp nguyên văn và trang trước khi chạy."
            ),
            "selection_criterion": (
                "Chọn trên dev theo Evidence Hit@3, tie-break bằng Hit@1; "
                "grounding/coherence thủ công là bước xác nhận. Test chỉ báo cáo sau khi khóa lựa chọn."
            ),
            "retrieval_results": table,
            "manual_results": manual_results,
            "trace_and_audit": (
                "Trace: `results/top3_trace.jsonl`; nhãn coherence: "
                "`results/coherence_audit.csv`; metadata/grounding: "
                "`results/metadata_grounding_audit.csv`. Precheck metadata/evidence "
                "tự động: `results/metadata_precheck.csv` và "
                "`results/grounding_precheck.csv`; inventory source unit: "
                "`results/source_unit_inventory.csv`. Các precheck không thay nhãn "
                "human; gợi ý Gemini (nếu chạy) nằm ở `results/*_ai_suggestions.csv` "
                "và không phải nhãn cuối. Coherence dùng 10 source-unit "
                "page đầu theo thứ tự; hướng dẫn chấm: `results/manual_review.md`; "
                "cấu hình/giả thuyết: `artifacts/experiment_log.csv`."
                if traces
                else "Benchmark chưa chạy; trace và nhãn audit chưa có."
            ),
            "analysis": analysis,
            "conclusion": _dev_candidate(summary),
            "limitations": (
                "Test chỉ có 5 câu; corpus là một tác phẩm tiếng Anh; "
                "nhãn coherence/metadata/grounding vẫn cần người chấm. "
                "Chi phí API không được suy đoán."
            ),
            "next_steps": next_steps,
        }
    )


def _write_audit_report_data(
    output_dir: Path,
    pages: list[DocumentPage],
    questions: list[dict[str, Any]],
    chunks_by_strategy: dict[str, list[Chunk]],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = output_dir / "benchmark_resolved.json"
    _write_json(resolved_path, questions)
    stats = _chunk_stats(chunks_by_strategy)
    _write_json(output_dir / "chunk_stats.json", stats)
    write_source_unit_inventory(output_dir, pages)
    return stats


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="download/validate data and prepare chunks without API or database calls",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    load_dotenv(STARTER_ROOT / ".env")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    manifest = _read_json(DATA_ROOT / "manifest.json")
    pages = _load_corpus(manifest)
    questions = _load_benchmark(BENCHMARK_ROOT / "questions.json", pages)
    chunks_by_strategy = _build_strategies(pages)
    chunk_stats = _write_audit_report_data(
        RESULTS_ROOT, pages, questions, chunks_by_strategy
    )

    model_name = os.getenv("EMBEDDING_MODEL") or "gemini-embedding-2"
    try:
        dimensions = int(os.getenv("EMBEDDING_DIMENSIONS", "768"))
    except ValueError as exc:
        raise ValueError("EMBEDDING_DIMENSIONS must be an integer") from exc
    experiment_log = _write_experiment_log(model_name, chunk_stats)

    report = ReportWriter(
        REPO_ROOT / "REPORT_TEMPLATE.md",
        REPORT_PATH,
    )
    _write_report(
        report,
        manifest,
        pages,
        questions,
        chunk_stats,
        model_name,
        dimensions,
    )

    logger.info(
        "Prepared %d source units, %d questions, and chunks A/B/C = %d/%d/%d",
        len(pages),
        len(questions),
        chunk_stats["A"]["chunk_count"],
        chunk_stats["B"]["chunk_count"],
        chunk_stats["C"]["chunk_count"],
    )
    logger.info("Experiment configuration logged at %s", experiment_log)
    if args.prepare_only:
        logger.info("Preparation complete; no API key or database was used")
        return

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is not set; configure starter/.env")
    embedder = MyEmbedder(model_name=model_name, dimensions=dimensions)
    store = PgVectorStore(database_url, dimensions)
    store.create_schema()
    database = store.describe()
    _write_report(
        report,
        manifest,
        pages,
        questions,
        chunk_stats,
        model_name,
        dimensions,
        database=database,
    )

    strategy_elapsed: dict[str, float] = {}
    vector_by_strategy: dict[str, list[list[float]]] = {}
    for strategy in ("A", "B"):
        start = time.perf_counter()
        vectors = embedder.embed_documents(
            [chunk.text for chunk in chunks_by_strategy[strategy]]
        )
        vector_by_strategy[strategy] = vectors
        store.upsert(strategy, chunks_by_strategy[strategy], vectors)
        strategy_elapsed[strategy] = time.perf_counter() - start
        _write_report(
            report,
            manifest,
            pages,
            questions,
            chunk_stats,
            embedder.model_name,
            embedder.dimensions,
            elapsed_by_strategy=strategy_elapsed,
            database=database,
            request_count=embedder.request_count,
            input_tokens=embedder.input_token_count,
        )
    vector_by_strategy["C"] = vector_by_strategy["B"]
    start = time.perf_counter()
    store.upsert("C", chunks_by_strategy["C"], vector_by_strategy["C"])
    strategy_elapsed["C"] = time.perf_counter() - start
    _write_report(
        report,
        manifest,
        pages,
        questions,
        chunk_stats,
        embedder.model_name,
        embedder.dimensions,
        elapsed_by_strategy=strategy_elapsed,
        database=database,
        request_count=embedder.request_count,
        input_tokens=embedder.input_token_count,
    )

    query_embedding_start = time.perf_counter()
    query_vectors = {
        question["id"]: embedder.embed_query(question["question"])
        for question in questions
    }
    query_embedding_seconds = time.perf_counter() - query_embedding_start
    _write_report(
        report,
        manifest,
        pages,
        questions,
        chunk_stats,
        embedder.model_name,
        embedder.dimensions,
        elapsed_by_strategy=strategy_elapsed,
        database=database,
        request_count=embedder.request_count,
        input_tokens=embedder.input_token_count,
        query_embedding_seconds=query_embedding_seconds,
    )
    traces: list[dict[str, Any]] = []
    search_seconds = defaultdict(float)
    for strategy in STRATEGY_ORDER:
        strategy_traces = []
        for question in questions:
            filters = question["filter"] if strategy == "C" else None
            start = time.perf_counter()
            hits = store.search(
                strategy,
                query_vectors[question["id"]],
                top_k=3,
                filters=filters,
            )
            search_seconds[strategy] += time.perf_counter() - start
            trace = evaluate_query(question, hits)
            trace["strategy"] = strategy
            traces.append(trace)
            strategy_traces.append(trace)
        strategy_elapsed[strategy] += search_seconds[strategy]
        current_summary = summarize(traces)
        _write_jsonl(RESULTS_ROOT / "top3_trace.jsonl", traces)
        _write_json(RESULTS_ROOT / "retrieval_metrics.json", current_summary)
        _write_experiment_log(model_name, chunk_stats, current_summary)
        _write_report(
            report,
            manifest,
            pages,
            questions,
            chunk_stats,
            embedder.model_name,
            embedder.dimensions,
            traces,
            current_summary,
            strategy_elapsed,
            database,
            embedder.request_count,
            embedder.input_token_count,
            query_embedding_seconds=query_embedding_seconds,
        )

    summary = summarize(traces)
    experiment_log = _write_experiment_log(model_name, chunk_stats, summary)
    output_dir = RESULTS_ROOT
    _write_jsonl(output_dir / "top3_trace.jsonl", traces)
    _write_json(output_dir / "retrieval_metrics.json", summary)
    audit_paths = _write_audits(
        output_dir,
        pages,
        chunks_by_strategy,
        [question for question in questions if question["split"] == "test"],
        traces,
    )
    run_metadata = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "embedding_model": embedder.model_name,
        "dimensions": embedder.dimensions,
        "embedding_requests": embedder.request_count,
        "input_tokens_reported_by_api": embedder.input_token_count,
        "cost_usd": None,
        "cost_note": "Check Google AI Studio billing; no price inferred by the runner.",
        "database": database,
        "exact_search_metric": "cosine",
        "elapsed_seconds": strategy_elapsed,
        "search_seconds": dict(search_seconds),
    }
    _write_json(output_dir / "run_metadata.json", run_metadata)
    _write_report(
        report,
        manifest,
        pages,
        questions,
        chunk_stats,
        embedder.model_name,
        embedder.dimensions,
        traces,
        summary,
        strategy_elapsed,
        database,
        embedder.request_count,
        embedder.input_token_count,
        audit_paths,
        query_embedding_seconds,
    )
    logger.info(
        "Pipeline complete. Results saved under %s; experiment log at %s",
        output_dir,
        experiment_log,
    )


if __name__ == "__main__":
    main()
