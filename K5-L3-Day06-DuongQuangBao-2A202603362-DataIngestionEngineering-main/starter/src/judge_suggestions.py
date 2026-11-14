"""Generate separate, human-reviewable Gemini audit suggestions."""
import argparse
import csv
import json
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

logger = logging.getLogger("audit_judge")
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "results"
PROMPT_VERSION = "day06-audit-rubric-v1"
COHERENCE_FIELDS = [
    "strategy",
    "page",
    "chunk_id",
    "ai_suggested_score",
    "ai_rationale",
    "ai_confidence",
    "judge_model",
    "prompt_version",
    "human_final_score",
    "human_rationale",
]
GROUNDING_FIELDS = [
    "case_id",
    "question_id",
    "strategy",
    "ai_suggested_score",
    "ai_rationale",
    "ai_confidence",
    "judge_model",
    "prompt_version",
    "human_final_score",
    "human_rationale",
]


def _batches(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _parse_scores(response: object, expected_ids: set[str]) -> dict[str, dict[str, Any]]:
    response_text = getattr(response, "text", None)
    if not isinstance(response_text, str) or not response_text.strip():
        raise RuntimeError("Gemini audit judge returned an empty response")
    response_text = response_text.strip()
    if response_text.startswith("```"):
        response_text = response_text.split("\n", 1)[-1].strip()
        if response_text.endswith("```"):
            response_text = response_text[:-3].strip()
    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Gemini audit judge returned invalid JSON") from exc
    scores = (
        payload.get("scores")
        if isinstance(payload, dict)
        else payload if isinstance(payload, list) else None
    )
    if not isinstance(scores, list):
        raise RuntimeError(
            "Gemini audit judge response must contain a scores array; "
            f"response={response_text[:1000]!r}"
        )

    parsed: dict[str, dict[str, Any]] = {}
    for row in scores:
        if not isinstance(row, dict):
            raise RuntimeError("Gemini audit judge returned a malformed score row")
        case_id = row.get("case_id")
        score = row.get("score")
        confidence = row.get("confidence")
        rationale = row.get("rationale")
        if (
            not isinstance(case_id, str)
            or case_id not in expected_ids
            or case_id in parsed
            or isinstance(score, bool)
            or not isinstance(score, int)
            or score not in {0, 1, 2}
            or not isinstance(confidence, str)
            or confidence not in {"low", "medium", "high"}
            or not isinstance(rationale, str)
            or not rationale.strip()
        ):
            raise RuntimeError("Gemini audit judge returned an invalid score row")
        parsed[case_id] = {
            "score": score,
            "rationale": rationale.strip(),
            "confidence": confidence,
        }
    if set(parsed) != expected_ids:
        missing = sorted(expected_ids - set(parsed))
        extra = sorted(set(parsed) - expected_ids)
        raise RuntimeError(
            f"Gemini audit judge case IDs mismatch; missing={missing}, extra={extra}"
        )
    return parsed


def _generate_batch(
    client: Any,
    model: str,
    instructions: str,
    cases: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    from google.genai import types

    prompt = (
        f"{instructions}\n\n"
        "Return only JSON with this shape: "
        '{"scores":[{"case_id":"...","score":0,"rationale":"...","confidence":"low|medium|high"}]}.\n'
        "Use each supplied case_id exactly once. Scores must be integer 0, 1, or 2. "
        "Do not include Markdown fences.\n\n"
        f"Cases:\n{json.dumps(cases, ensure_ascii=False)}"
    )
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0),
    )
    return _parse_scores(
        response,
        {str(case["case_id"]) for case in cases},
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _write_rows(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def _coherence_suggestions(client: Any, model: str) -> Path:
    audit_rows = _read_csv(RESULTS_ROOT / "coherence_audit.csv")
    text_by_chunk: dict[str, str] = {}
    for row in audit_rows:
        prior_text = text_by_chunk.setdefault(row["chunk_id"], row["chunk_text"])
        if prior_text != row["chunk_text"]:
            raise ValueError(f"Chunk ID has conflicting text: {row['chunk_id']}")
    cases = [
        {"case_id": chunk_id, "chunk_text": text}
        for chunk_id, text in text_by_chunk.items()
    ]
    instructions = (
        "You are proposing draft coherence ratings for a human reviewer. "
        "Judge only whether each chunk is understandable by itself and preserves "
        "a complete idea. Score 0 if cut mid-idea or unintelligible, 1 if partly "
        "understandable but missing context, and 2 if self-contained and coherent. "
        "Do not score retrieval relevance or factual correctness. Use concise, "
        "text-grounded rationales; choose lower confidence when ambiguous."
    )
    suggestions: dict[str, dict[str, Any]] = {}
    for batch in _batches(cases, 5):
        suggestions.update(_generate_batch(client, model, instructions, batch))
        logger.info(
            "Received coherence suggestions for %d/%d chunks",
            len(suggestions),
            len(cases),
        )

    output_rows = []
    for source_row in audit_rows:
        case_id = source_row["chunk_id"]
        suggestion = suggestions[case_id]
        output_rows.append(
            {
                "strategy": source_row["strategy"],
                "page": source_row["page"],
                "chunk_id": case_id,
                "ai_suggested_score": suggestion["score"],
                "ai_rationale": suggestion["rationale"],
                "ai_confidence": suggestion["confidence"],
                "judge_model": model,
                "prompt_version": PROMPT_VERSION,
                "human_final_score": "",
                "human_rationale": "",
            }
        )
    output_path = RESULTS_ROOT / "coherence_ai_suggestions.csv"
    _write_rows(output_path, COHERENCE_FIELDS, output_rows)
    return output_path


def _grounding_suggestions(client: Any, model: str) -> Path:
    benchmark = json.loads(
        (REPO_ROOT / "benchmark" / "questions.json").read_text(encoding="utf-8")
    )
    questions = {question["id"]: question for question in benchmark}
    traces = [
        json.loads(line)
        for line in (RESULTS_ROOT / "top3_trace.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    test_traces = [trace for trace in traces if trace["split"] == "test"]
    cases: list[dict[str, Any]] = []
    source_by_case: dict[str, tuple[str, str]] = {}
    for index, trace in enumerate(test_traces, start=1):
        question = questions[trace["question_id"]]
        case_id = f"grounding-{index:02d}"
        source_by_case[case_id] = (trace["question_id"], trace["strategy"])
        cases.append(
            {
                "case_id": case_id,
                "question": trace["question"],
                "expected_answer": question["expected_answer"],
                "gold_evidence": trace["evidence_quote"],
                "retrieved_top3": [
                    {
                        "rank": rank,
                        "source": hit["source"],
                        "page": hit["page"],
                        "text": hit["text"],
                    }
                    for rank, hit in enumerate(trace["hits"], start=1)
                ],
            }
        )
    instructions = (
        "You are proposing draft evidence-grounding ratings for a human reviewer. "
        "Judge the combined retrieved top-3 only; do not use outside knowledge. "
        "Score 0 if it provides no usable evidence, 1 if it has relevant evidence "
        "but misses a needed condition or context, and 2 if it is sufficient to "
        "support the expected answer and traceable to a source. This evaluates "
        "retrieved evidence, not an LLM-generated answer. Use concise rationales "
        "that cite retrieved ranks and note missing context; do not infer evidence "
        "that is absent."
    )
    suggestions: dict[str, dict[str, Any]] = {}
    for batch in _batches(cases, 5):
        suggestions.update(_generate_batch(client, model, instructions, batch))
        logger.info(
            "Received grounding suggestions for %d/%d question-strategy cases",
            len(suggestions),
            len(cases),
        )

    output_rows = []
    for case in cases:
        case_id = case["case_id"]
        suggestion = suggestions[case_id]
        question_id, strategy = source_by_case[case_id]
        output_rows.append(
            {
                "case_id": case_id,
                "question_id": question_id,
                "strategy": strategy,
                "ai_suggested_score": suggestion["score"],
                "ai_rationale": suggestion["rationale"],
                "ai_confidence": suggestion["confidence"],
                "judge_model": model,
                "prompt_version": PROMPT_VERSION,
                "human_final_score": "",
                "human_rationale": "",
            }
        )
    output_path = RESULTS_ROOT / "grounding_ai_suggestions.csv"
    _write_rows(output_path, GROUNDING_FIELDS, output_rows)
    return output_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=os.getenv("AUDIT_JUDGE_MODEL", "gemini-2.5-flash"),
        help="Gemini text model used only for draft audit suggestions",
    )
    args = parser.parse_args(argv)
    load_dotenv(REPO_ROOT / "starter" / ".env")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not os.getenv("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is not set")

    from google import genai

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    paths = [
        _coherence_suggestions(client, args.model),
        _grounding_suggestions(client, args.model),
    ]
    for path in paths:
        logger.info("Draft suggestions saved at %s; human review is still required", path)


if __name__ == "__main__":
    main()
