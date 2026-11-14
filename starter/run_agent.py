"""Run real Gemini tool-calling agents with Day 7 retrieval methods."""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from agentic import run_agent
from baseline import hit, pack_context, validate
from benchmark import validate_minimum_benchmark
from gemini import GeminiController, GeminiEmbedder
from retrieval import DenseRetriever, HybridRetriever, LexicalRetriever


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def _rate(n: int, total: int) -> dict[str, Any]:
    return {'n': n, 'N': total, 'rate': n / total if total else None}


def _usage_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    return {key: after[key] - before[key] for key in after}


def select_questions(
    questions: list[dict[str, Any]],
    split: str,
    question_id: str | None = None,
) -> list[dict[str, Any]]:
    selected = [question for question in questions if question['split'] == split]
    if question_id is None:
        return selected
    matches = [question for question in selected if question['id'] == question_id]
    if not matches:
        raise ValueError(f'Không tìm thấy question ID {question_id!r} trong split {split}')
    return matches


def _build_retrievers(
    methods: list[str],
    chunks: list[dict[str, Any]],
    *,
    database_url: str,
    embedder: GeminiEmbedder | None,
    strategy: str,
    candidate_k: int,
) -> dict[str, Any]:
    lexical = LexicalRetriever(chunks) if {'lexical', 'hybrid'} & set(methods) else None
    dense = None
    if {'dense', 'hybrid'} & set(methods):
        if embedder is None:
            raise RuntimeError('Dense retrieval cần Gemini Embedding 2')
        dense = DenseRetriever(database_url, embedder, strategy=strategy)
    retrievers = {}
    if 'dense' in methods:
        retrievers['dense'] = dense
    if 'lexical' in methods:
        retrievers['lexical'] = lexical
    if 'hybrid' in methods:
        retrievers['hybrid'] = HybridRetriever(dense, lexical, candidate_k=candidate_k)
    return retrievers


def run_method(
    method: str,
    retriever: Any,
    questions: list[dict[str, Any]],
    *,
    agent_model: str | None,
    top_k: int,
    max_searches: int,
    budget: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    traces = []
    started_all = time.perf_counter()
    controller = GeminiController(model=agent_model)
    answerable = [question for question in questions if question.get('answerable', True)]
    hit1_count = 0
    hit3_count = 0
    context_hit_count = 0
    tool_use_count = 0
    successful_count = 0

    for question in questions:
        started = time.perf_counter()
        answerable_case = question.get('answerable', True)
        usage_before = controller.usage_snapshot()
        original_results = []
        retrieval_hit1 = None
        retrieval_hit3 = None
        try:
            original_results = retriever.search(
                question['question'],
                filters=question.get('filter'),
                top_k=3,
            )
            if answerable_case:
                gold = question.get('gold', [])
                retrieval_hit1 = bool(original_results and hit(original_results[0], gold))
                retrieval_hit3 = any(hit(row, gold) for row in original_results)
                hit1_count += retrieval_hit1
                hit3_count += retrieval_hit3
            result = run_agent(
                question['question'],
                controller,
                retriever,
                filters=question.get('filter'),
                max_searches=max_searches,
                top_k=top_k,
                budget=budget,
            )
            events = result['events']
            last_search = next(
                (event for event in reversed(events) if event.get('event') == 'search'),
                None,
            )
            context_ids = last_search['context_ids'] if last_search else []
            if answerable_case:
                context_hit_count += any(
                    row['id'] in context_ids and hit(row, question.get('gold', []))
                    for row in original_results
                )
            if any(event.get('event') == 'search' for event in events):
                tool_use_count += 1
            successful_count += 1
            trace = {
                'query_id': question['id'],
                'split': question['split'],
                'question': question['question'],
                'answerable': answerable_case,
                'method': method,
                'status': result['status'],
                'answer': result['answer'],
                'citations': result['citations'],
                'events': events,
                'original_query_top3': original_results,
                'context_ids': context_ids,
                'hit1': retrieval_hit1,
                'hit3': retrieval_hit3,
                'usage': _usage_delta(usage_before, controller.usage_snapshot()),
            }
        except Exception as exc:
            print(
                f"[{method}] {question['id']}: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            trace = {
                'query_id': question['id'],
                'split': question['split'],
                'question': question['question'],
                'answerable': answerable_case,
                'method': method,
                'status': 'error',
                'answer': '',
                'citations': [],
                'events': [],
                'original_query_top3': original_results,
                'context_ids': [],
                'hit1': retrieval_hit1,
                'hit3': retrieval_hit3,
                'error': {'type': type(exc).__name__, 'message': str(exc)},
                'usage': _usage_delta(usage_before, controller.usage_snapshot()),
            }
        trace['elapsed_ms'] = (time.perf_counter() - started) * 1000
        traces.append(trace)

    n_answerable = len(answerable)
    n_unanswerable = len(questions) - n_answerable
    elapsed_values = [trace['elapsed_ms'] for trace in traces]
    search_counts = [
        sum(event.get('event') == 'search' for event in trace['events'])
        for trace in traces if trace['status'] != 'error'
    ]
    errors = [trace for trace in traces if trace['status'] == 'error']
    summary = {
        'method': method,
        'split': questions[0]['split'] if questions else None,
        'total_questions': len(questions),
        'answerable_questions': n_answerable,
        'unanswerable_questions': n_unanswerable,
        'Hit@1': _rate(hit1_count, n_answerable),
        'Evidence_Hit@3': _rate(hit3_count, n_answerable),
        'Context_Evidence_Rate': _rate(context_hit_count, n_answerable),
        'tool_use': _rate(tool_use_count, successful_count),
        'search_calls': {
            'total': sum(search_counts),
            'mean_per_successful_case': statistics.mean(search_counts) if search_counts else None,
            'distribution': {str(value): search_counts.count(value) for value in sorted(set(search_counts))},
        },
        'configuration': {
            'top_k': top_k,
            'max_searches': max_searches,
            'context_budget_chars': budget,
            'candidate_k': getattr(retriever, 'candidate_k', None),
        },
        'errors': [{'query_id': trace['query_id'], **trace['error']} for trace in errors],
        'timing_ms': {
            'median_per_case': statistics.median(elapsed_values) if elapsed_values else None,
            'mean_per_case': statistics.mean(elapsed_values) if elapsed_values else None,
            'total': (time.perf_counter() - started_all) * 1000,
        },
        'agent_model': agent_model or os.getenv('GEMINI_AGENT_MODEL', 'gemini-2.5-flash-lite'),
        'embedding_billing': 'Gemini Embedding 2 free tier (reported for Day 6)',
        'agent_cost_usd': None,
        'cost_note': 'Agent cost is not estimated; check current Gemini project billing.',
    }
    return traces, summary


def main() -> None:
    parser = argparse.ArgumentParser(description='Chạy agent Day 7 và lưu trace theo từng câu')
    parser.add_argument('--chunks', required=True, type=Path)
    parser.add_argument('--questions', required=True, type=Path)
    parser.add_argument('--split', required=True, choices=('dev', 'test'))
    parser.add_argument('--question-id', help='Chỉ chạy một câu hỏi thuộc split đã chọn')
    parser.add_argument('--methods', default='dense,lexical,hybrid')
    parser.add_argument('--strategy', default=os.getenv('DAY6_STRATEGY', 'C'))
    parser.add_argument('--dimensions', type=int, default=768)
    parser.add_argument('--agent-model', default=None)
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--candidate-k', type=int, default=20)
    parser.add_argument('--max-searches', type=int, default=4)
    parser.add_argument('--budget', type=int, default=6000, help='context budget tính bằng ký tự')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    chunks = _read_json(args.chunks)
    all_questions = _read_json(args.questions)
    counts = validate_minimum_benchmark(chunks, all_questions)
    selected = select_questions(all_questions, args.split, args.question_id)
    validate(chunks, selected)
    methods = [method.strip() for method in args.methods.split(',') if method.strip()]
    supported = {'dense', 'lexical', 'hybrid'}
    if not methods or len(set(methods)) != len(methods) or set(methods) - supported:
        raise ValueError(f'--methods phải là danh sách không lặp từ {sorted(supported)}')
    if args.top_k <= 0 or args.candidate_k <= 0 or args.max_searches <= 0 or args.budget <= 0:
        raise ValueError('top-k, candidate-k, max-searches và budget phải lớn hơn 0')

    database_url = os.getenv('DATABASE_URL', '')
    needs_dense = bool({'dense', 'hybrid'} & set(methods))
    embedder = GeminiEmbedder(dimensions=args.dimensions) if needs_dense else None
    retrievers = _build_retrievers(
        methods,
        chunks,
        database_url=database_url,
        embedder=embedder,
        strategy=args.strategy,
        candidate_k=args.candidate_k,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    suffix = f'{args.split}-{args.question_id}' if args.question_id else args.split
    cases_path = args.output / f'cases-{suffix}.json'
    output_paths = [cases_path] + [
        path
        for method in methods
        for path in (
            args.output / f'{method}-{suffix}-traces.json',
            args.output / f'{method}-{suffix}-summary.json',
        )
    ]
    existing = [path for path in output_paths if path.exists()]
    if existing:
        raise FileExistsError(
            'Output đã tồn tại, chọn thư mục khác để không ghi đè: '
            + ', '.join(str(path) for path in existing)
        )
    cases_path.write_text(
        json.dumps(
            [
                {
                    'id': question['id'],
                    'split': question['split'],
                    'answerable': question.get('answerable', True),
                }
                for question in selected
            ],
            ensure_ascii=False,
            indent=2,
        ) + '\n',
        encoding='utf-8',
    )
    for method in methods:
        requests_before = embedder.request_count if method in ('dense', 'hybrid') else 0
        tokens_before = embedder.input_tokens if method in ('dense', 'hybrid') else None
        traces, summary = run_method(
            method,
            retrievers[method],
            selected,
            agent_model=args.agent_model,
            top_k=args.top_k,
            max_searches=args.max_searches,
            budget=args.budget,
        )
        summary['benchmark_counts'] = counts[args.split]
        summary['question_id'] = args.question_id
        summary['strategy'] = args.strategy if method in ('dense', 'hybrid') else None
        summary['embedding_model'] = embedder.model if method in ('dense', 'hybrid') else None
        summary['embedding_dimensions'] = embedder.dimensions if method in ('dense', 'hybrid') else None
        summary['embedding_requests'] = (
            embedder.request_count - requests_before if method in ('dense', 'hybrid') else 0
        )
        if method in ('dense', 'hybrid'):
            summary['embedding_tokens_reported'] = (
                embedder.input_tokens - tokens_before
                if embedder.input_tokens is not None and tokens_before is not None
                else embedder.input_tokens
            )
        else:
            summary['embedding_tokens_reported'] = None
        (args.output / f'{method}-{suffix}-traces.json').write_text(
            json.dumps(traces, ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8',
        )
        (args.output / f'{method}-{suffix}-summary.json').write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + '\n',
            encoding='utf-8',
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
