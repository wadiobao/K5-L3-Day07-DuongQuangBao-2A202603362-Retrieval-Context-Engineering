import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'starter'))
from run_agent import run_method, select_questions


CHUNK = {
    'id': 'c1',
    'source_id': 'doc',
    'source': 'source.txt',
    'page': 1,
    'text': 'The answer is supported.',
    'metadata': {},
    'score': 0.9,
}


class FakeController:
    def __init__(self, model=None, min_request_interval_seconds=0.0):
        self.calls = 0

    def usage_snapshot(self):
        return {'calls': self.calls, 'total_tokens': 0}

    def decide(self, question, history):
        self.calls += 1
        if not history:
            return {'action': 'search', 'query': question}
        if 'missing' in question:
            return {'action': 'answer', 'answer': 'Không đủ bằng chứng.', 'citations': []}
        return {'action': 'answer', 'answer': 'Được hỗ trợ [c1].', 'citations': ['c1']}


class FakeRetriever:
    def search(self, query, *, filters, top_k):
        return [CHUNK][:top_k]


class BrokenController:
    def __init__(self, model=None, min_request_interval_seconds=0.0):
        pass

    def usage_snapshot(self):
        return {'calls': 0, 'total_tokens': 0}

    def decide(self, question, history):
        raise RuntimeError('agent quota exceeded')


class AgentRunnerTests(unittest.TestCase):
    def test_selects_one_question_from_the_requested_split(self):
        questions = [
            {'id': 'q01', 'split': 'dev'},
            {'id': 'q02', 'split': 'dev'},
            {'id': 'q03', 'split': 'test'},
        ]
        self.assertEqual(select_questions(questions, 'dev', 'q02'), [questions[1]])
        with self.assertRaisesRegex(ValueError, 'split dev'):
            select_questions(questions, 'dev', 'q03')

    def test_summary_excludes_missing_evidence_from_retrieval_denominator(self):
        questions = [
            {
                'id': 'q-answerable',
                'split': 'dev',
                'question': 'What is the answer?',
                'answerable': True,
                'gold': [{'source_id': 'doc', 'page': 1, 'evidence': 'supported'}],
            },
            {
                'id': 'q-missing',
                'split': 'dev',
                'question': 'What is missing?',
                'answerable': False,
                'expected_behavior': 'search_and_abstain',
                'absence_reason': 'Not in corpus.',
            },
        ]
        with patch('run_agent.GeminiController', FakeController):
            traces, summary = run_method(
                'lexical',
                FakeRetriever(),
                questions,
                agent_model='test-model',
                top_k=3,
                max_searches=2,
                budget=800,
                agent_cost_per_case_usd=0.0,
            )
        self.assertTrue(traces[1]['status'] == 'abstain')
        self.assertEqual(summary['Hit@1'], {'n': 1, 'N': 1, 'rate': 1.0})
        self.assertEqual(summary['Context_Evidence_Rate'], {'n': 1, 'N': 1, 'rate': 1.0})
        self.assertEqual(summary['unanswerable_questions'], 1)
        self.assertEqual(summary['agent_cost_usd'], 0.0)
        self.assertEqual(traces[0]['agent_cost_usd'], 0.0)
        self.assertEqual(traces[1]['agent_cost_usd'], 0.0)

    def test_agent_errors_do_not_erase_retrieval_measurements(self):
        question = {
            'id': 'q-answerable',
            'split': 'dev',
            'question': 'What is the answer?',
            'answerable': True,
            'gold': [{'source_id': 'doc', 'page': 1, 'evidence': 'supported'}],
        }
        with patch('run_agent.GeminiController', BrokenController):
            traces, summary = run_method(
                'lexical',
                FakeRetriever(),
                [question],
                agent_model='test-model',
                top_k=3,
                max_searches=2,
                budget=800,
            )
        self.assertEqual(traces[0]['status'], 'error')
        self.assertEqual(traces[0]['original_query_top3'], [CHUNK])
        self.assertTrue(traces[0]['hit1'] and traces[0]['hit3'])
        self.assertEqual(summary['Hit@1'], {'n': 1, 'N': 1, 'rate': 1.0})
        self.assertEqual(summary['Context_Evidence_Rate'], {'n': 0, 'N': 1, 'rate': 0.0})


if __name__ == '__main__':
    unittest.main()
