"""Behavior checks for the runnable Day 7 baseline and Day 6 JSON adapter."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'starter'))
from baseline import TfidfIndex, evaluate, hit, pack_context, validate
from convert_day6 import convert_chunk, convert_question


def chunk(id, text, page=1):
    return {'id': id, 'source_id': 'doc', 'source': 'doc.pdf', 'page': page,
            'text': text, 'metadata': {'year': 2025}}


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [chunk('a', 'Hoàn tiền mất bảy ngày.'), chunk('b', 'Giao hàng mất ba ngày.', 2)]
        self.q = {'id': 'q', 'split': 'dev', 'question': 'Hoàn tiền bao lâu?',
                  'gold': [{'source_id': 'doc', 'page': 1, 'evidence': 'bảy ngày'}], 'expected_answer': 'bảy ngày'}

    def test_rank_and_evidence(self):
        validate(self.chunks, [self.q])
        trace = evaluate(TfidfIndex(self.chunks), [self.q])[0]
        self.assertEqual(trace['results'][0]['id'], 'a')
        self.assertTrue(trace['hit1'] and trace['hit3'] and trace['context_hit'])
        self.assertFalse(hit(self.chunks[1], self.q['gold']))

    def test_gold_alternative_and_wrong_page(self):
        valid = {'source_id': 'doc', 'page': 1, 'evidence': 'bảy ngày'}
        wrong = {'source_id': 'doc', 'page': 2, 'evidence': 'bảy ngày'}
        self.assertFalse(hit(self.chunks[0], [wrong]))
        self.assertTrue(hit(self.chunks[0], [wrong, valid]))
        with self.assertRaises(ValueError):
            validate(self.chunks, [{**self.q, 'gold': [wrong]}])

    def test_budget_dedup_and_empty_query(self):
        context, ids = pack_context([self.chunks[0], self.chunks[0]], 800)
        self.assertEqual(ids, ['a'])
        self.assertLessEqual(len(context), 800)
        self.assertEqual(pack_context(self.chunks, 1), ('', []))
        self.assertEqual(TfidfIndex(self.chunks).search('khôngkhớp'), [])
        trace = evaluate(TfidfIndex(self.chunks), [self.q], budget=1)[0]
        self.assertTrue(trace['hit3'])
        self.assertFalse(trace['context_hit'])

    def test_unanswerable_questions_are_valid_and_excluded_from_retrieval_metrics(self):
        unanswerable = {
            'id': 'q-missing',
            'split': 'dev',
            'question': 'Trong tài liệu có quy định về hoàn tiền bằng vàng không?',
            'answerable': False,
            'expected_behavior': 'search_and_abstain',
            'absence_reason': 'Đã rà soát nội dung và không có quy định này.',
        }
        validate(self.chunks, [unanswerable])
        trace = evaluate(TfidfIndex(self.chunks), [unanswerable])[0]
        self.assertIsNone(trace['hit1'])
        self.assertIsNone(trace['hit3'])
        self.assertIsNone(trace['context_hit'])

        with self.assertRaises(ValueError):
            validate(self.chunks, [{**unanswerable, 'gold': [{'source_id': 'doc', 'page': 1, 'evidence': 'bảy ngày'}]}])

    def test_lexical_retrieval_respects_metadata_filters(self):
        filtered = TfidfIndex([
            {**self.chunks[0], 'metadata': {'section_id': 'chapter-1'}},
            {**self.chunks[1], 'metadata': {'section_id': 'chapter-2'}},
        ])
        self.assertEqual(
            [row['id'] for row in filtered.search('ngày', filters={'section_id': 'chapter-2'})],
            ['b'],
        )

    def test_convert_day6_and_preserve_filter(self):
        raw_chunk = {'chunk_id': 'c1', 'document_id': 'doc', 'source': 'doc.pdf', 'page': 1,
                     'text': 'bảy ngày', 'metadata': {'year': 2025}}
        raw_q = {'id': 'q', 'split': 'dev', 'question': 'bao lâu', 'gold_source_id': 'doc',
                 'gold_page': 1, 'evidence_quote': 'bảy ngày', 'expected_answer': 'bảy ngày', 'filter': {'year': 2025}}
        c, q = convert_chunk(raw_chunk), convert_question(raw_q)
        validate([c], [q])
        self.assertEqual(q['filter'], {'year': 2025})
        self.assertEqual(c['id'], 'c1')

    def test_convert_unanswerable_question(self):
        raw = {
            'id': 'q-missing',
            'split': 'dev',
            'question': 'Không có trong corpus?',
            'answerable': False,
            'absence_reason': 'Đã rà soát corpus.',
        }
        self.assertEqual(
            convert_question(raw),
            {
                'id': 'q-missing',
                'split': 'dev',
                'question': 'Không có trong corpus?',
                'answerable': False,
                'expected_behavior': 'search_and_abstain',
                'absence_reason': 'Đã rà soát corpus.',
            },
        )

    def test_cli_uses_requested_split_only(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root/'chunks.json').write_text(json.dumps(self.chunks), encoding='utf-8')
            # A test gold which fails validation must not be inspected by a dev run.
            test = {**self.q, 'id': 't', 'split': 'test', 'gold': [{'source_id': 'x', 'page': 9, 'evidence': 'missing'}]}
            (root/'q.json').write_text(json.dumps([self.q, test]), encoding='utf-8')
            subprocess.run([sys.executable, str(ROOT/'starter/baseline.py'), '--chunks', str(root/'chunks.json'),
                            '--questions', str(root/'q.json'), '--split', 'dev', '--output', str(root/'out')],
                           check=True, capture_output=True, text=True)
            summary = json.loads((root/'out/summary.json').read_text())
            self.assertEqual((summary['split'], summary['n'], summary['hit1']), ('dev', 1, 1))
            self.assertEqual(len(json.loads((root/'out/traces.json').read_text())), 1)


if __name__ == '__main__': unittest.main()
