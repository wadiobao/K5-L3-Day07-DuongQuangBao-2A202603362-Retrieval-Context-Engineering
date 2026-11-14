"""Validation for the minimum Day 7 dev/test benchmark sizes."""
from typing import Any

from baseline import validate


MINIMUMS = {
    'dev': {'answerable': 15, 'unanswerable': 2},
    'test': {'answerable': 5, 'unanswerable': 2},
}


def validate_minimum_benchmark(
    chunks: list[dict[str, Any]],
    questions: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    validate(chunks, questions)
    counts = {
        split: {
            'answerable': sum(
                q.get('answerable', True) is True for q in questions if q['split'] == split
            ),
            'unanswerable': sum(
                q.get('answerable', True) is False for q in questions if q['split'] == split
            ),
        }
        for split in ('dev', 'test')
    }
    for split, expected in MINIMUMS.items():
        for kind, minimum in expected.items():
            actual = counts[split][kind]
            if actual < minimum:
                raise ValueError(
                    f'{split} cần ít nhất {minimum} câu {kind}; hiện có {actual}'
                )
    return counts
