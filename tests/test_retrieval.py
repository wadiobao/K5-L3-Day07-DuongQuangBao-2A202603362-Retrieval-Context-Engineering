import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'starter'))
from retrieval import DenseRetriever, HybridRetriever, LexicalRetriever


def chunk(chunk_id, text):
    return {
        'id': chunk_id,
        'source_id': 'doc',
        'source': 'book.txt',
        'page': 1,
        'text': text,
        'metadata': {},
    }


class FakeEmbedder:
    dimensions = 768


class FakeDense:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def search(self, query, *, filters, top_k):
        self.calls.append((query, filters, top_k))
        return self.rows[:top_k]


class RetrievalTests(unittest.TestCase):
    def test_hybrid_uses_rrf_and_preserves_filters(self):
        rows = [
            {**chunk('a', 'quasar alpha'), 'metadata': {'edition': 'one'}},
            {**chunk('b', 'black hole'), 'metadata': {'edition': 'one'}},
        ]
        dense_rows = [{**rows[1], 'score': 0.9}, {**rows[0], 'score': 0.7}]
        dense = FakeDense(dense_rows)
        lexical = LexicalRetriever(rows)
        hybrid = HybridRetriever(dense, lexical, candidate_k=2)
        result = hybrid.search('quasar', filters={'edition': 'one'}, top_k=2)
        self.assertEqual({row['strategy'] for row in result}, {'hybrid_rrf'})
        self.assertEqual(dense.calls, [('quasar', {'edition': 'one'}, 2)])
        self.assertEqual([row['id'] for row in result], ['a', 'b'])
        self.assertGreater(result[0]['score'], result[1]['score'])

    def test_rejects_bad_dense_configuration(self):
        with self.assertRaises(ValueError):
            DenseRetriever('', FakeEmbedder())
        with self.assertRaises(ValueError):
            HybridRetriever(FakeDense([]), LexicalRetriever([chunk('a', 'x')]), candidate_k=0)


if __name__ == '__main__':
    unittest.main()
