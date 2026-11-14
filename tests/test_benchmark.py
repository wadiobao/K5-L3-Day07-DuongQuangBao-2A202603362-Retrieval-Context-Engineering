import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'starter'))
from benchmark import validate_minimum_benchmark


def make_dataset():
    chunks = []
    questions = []
    for split, answerable_count, unanswerable_count in (
        ('dev', 15, 2),
        ('test', 5, 2),
    ):
        for index in range(answerable_count):
            chunk_id = f'{split}-{index}'
            text = f'evidence-{chunk_id}'
            chunks.append({
                'id': chunk_id,
                'source_id': 'doc',
                'source': 'source.txt',
                'page': index + 1,
                'text': text,
                'metadata': {},
            })
            questions.append({
                'id': f'q-{chunk_id}',
                'split': split,
                'question': f'Question {chunk_id}?',
                'answerable': True,
                'expected_answer': text,
                'gold': [{'source_id': 'doc', 'page': index + 1, 'evidence': text}],
            })
        for index in range(unanswerable_count):
            questions.append({
                'id': f'q-{split}-missing-{index}',
                'split': split,
                'question': f'Missing question {index}?',
                'answerable': False,
                'expected_behavior': 'search_and_abstain',
                'absence_reason': 'Evidence is absent from the corpus.',
            })
    return chunks, questions


class BenchmarkTests(unittest.TestCase):
    def test_accepts_minimum_day7_counts(self):
        counts = validate_minimum_benchmark(*make_dataset())
        self.assertEqual(counts['dev'], {'answerable': 15, 'unanswerable': 2})
        self.assertEqual(counts['test'], {'answerable': 5, 'unanswerable': 2})

    def test_rejects_missing_abstention_cases(self):
        chunks, questions = make_dataset()
        questions.pop()
        with self.assertRaisesRegex(ValueError, 'test cần ít nhất 2 câu unanswerable'):
            validate_minimum_benchmark(chunks, questions)


if __name__ == '__main__':
    unittest.main()
