"""Gold-evidence retrieval metrics and human-audit sample preparation."""
from collections import defaultdict
from typing import Any

from .ingestion import normalize_text


def _gold_locations(question: dict[str, Any]) -> list[tuple[str, int]]:
    locations = question.get("gold_locations")
    if locations:
        return [
            (str(item["gold_source_id"]), int(item["gold_page"]))
            for item in locations
        ]
    return [
        (str(question["gold_source_id"]), int(question["gold_page"]))
    ]


def _is_gold_hit(
    question: dict[str, Any],
    hit: dict[str, Any],
    locations: list[tuple[str, int]],
) -> bool:
    source_id = str(hit.get("document_id", hit.get("metadata", {}).get("document_id", "")))
    page = hit.get("page", hit.get("metadata", {}).get("page"))
    if not any(source_id == gold_source and page == gold_page for gold_source, gold_page in locations):
        return False

    evidence = question.get("evidence_quote", "")
    if not evidence:
        return False
    normalized_evidence = normalize_text(evidence).casefold()
    normalized_chunk = normalize_text(hit.get("text", "")).casefold()
    return normalized_evidence in normalized_chunk


def evaluate_query(
    question: dict[str, Any],
    hits: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate top-k hits against verified source/page evidence."""
    locations = _gold_locations(question)
    hit_flags = [_is_gold_hit(question, hit, locations) for hit in hits]
    filters = question.get("filter", question.get("filters")) or {}
    filter_flags = [
        all(hit.get("metadata", {}).get(key) == value for key, value in filters.items())
        for hit in hits
    ]
    return {
        "question_id": question["id"],
        "split": question["split"],
        "question": question["question"],
        "filters": filters,
        "gold_locations": [
            {"gold_source_id": source_id, "gold_page": page}
            for source_id, page in locations
        ],
        "evidence_quote": question["evidence_quote"],
        "hit1": bool(hit_flags and hit_flags[0]),
        "evidence_hit3": any(hit_flags[:3]),
        "filter_compliant_hits": sum(filter_flags),
        "filter_checked_hits": len(hits) if filters else 0,
        "filter_compliant": bool(filters and hits and all(filter_flags)),
        "hits": hits[:3],
    }


def summarize(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate Hit@1 and Evidence Hit@3 as explicit counts and rates."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for trace in traces:
        grouped[(trace["strategy"], trace["split"])].append(trace)

    summary = []
    for strategy, split in sorted(grouped):
        rows = grouped[(strategy, split)]
        hit1_count = sum(bool(row["hit1"]) for row in rows)
        hit3_count = sum(bool(row["evidence_hit3"]) for row in rows)
        summary.append(
            {
                "strategy": strategy,
                "split": split,
                "question_count": len(rows),
                "hit1_count": hit1_count,
                "hit1_rate": hit1_count / len(rows),
                "evidence_hit3_count": hit3_count,
                "evidence_hit3_rate": hit3_count / len(rows),
            }
        )
    return summary
