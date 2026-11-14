import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'starter'))
from gemini import GeminiController, GeminiEmbedder


class FakeModels:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class GeminiControllerTests(unittest.TestCase):
    def test_function_call_is_normalized_and_usage_is_recorded(self):
        call = SimpleNamespace(name='search', args={'query': 'chapter 2'})
        response = SimpleNamespace(
            candidates=[SimpleNamespace(content=SimpleNamespace(parts=[
                SimpleNamespace(function_call=call),
            ]))],
            usage_metadata=SimpleNamespace(
                prompt_token_count=10,
                candidates_token_count=4,
                total_token_count=14,
            ),
        )
        models = FakeModels(response)
        controller = GeminiController(client=SimpleNamespace(models=models), model='test-model')
        result = controller.decide('What happens?', [{'context': '[c1] evidence'}])
        self.assertEqual(result, {'action': 'search', 'query': 'chapter 2'})
        self.assertEqual(controller.usage_snapshot(), {
            'calls': 1, 'prompt_tokens': 10, 'candidate_tokens': 4, 'total_tokens': 14,
        })
        self.assertEqual(models.calls[0]['model'], 'test-model')

    def test_controller_formats_model_selected_citation_ids(self):
        call = SimpleNamespace(
            name='answer',
            args={'answer': 'The answer is a watch.', 'citations': ['chunk-1']},
        )
        response = SimpleNamespace(
            candidates=[SimpleNamespace(content=SimpleNamespace(parts=[
                SimpleNamespace(function_call=call),
            ]))],
            usage_metadata=None,
        )
        controller = GeminiController(client=SimpleNamespace(models=FakeModels(response)))
        self.assertEqual(
            controller.decide('question', []),
            {
                'action': 'answer',
                'answer': 'The answer is a watch. [chunk-1]',
                'citations': ['chunk-1'],
            },
        )

    def test_missing_tool_call_is_an_explicit_error(self):
        response = SimpleNamespace(candidates=[])
        controller = GeminiController(client=SimpleNamespace(models=FakeModels(response)))
        with self.assertRaisesRegex(RuntimeError, 'không trả function call'):
            controller.decide('question', [])

    def test_embedding_matches_day6_model_and_dimensions(self):
        class FakeEmbeddingModels:
            def __init__(self):
                self.calls = []

            def embed_content(self, **kwargs):
                self.calls.append(kwargs)
                return SimpleNamespace(
                    embeddings=[SimpleNamespace(values=[0.25] * 768)],
                    usage_metadata=SimpleNamespace(total_token_count=3),
                )

        models = FakeEmbeddingModels()
        embedder = GeminiEmbedder(
            client=SimpleNamespace(models=models),
            model='gemini-embedding-2',
            dimensions=768,
        )
        vector = embedder.embed_query('How long?')
        self.assertEqual(len(vector), 768)
        self.assertEqual(models.calls[0]['model'], 'gemini-embedding-2')
        self.assertEqual(
            models.calls[0]['contents'],
            'task: search result | query: How long?',
        )
        self.assertEqual(embedder.request_count, 1)
        self.assertEqual(embedder.input_tokens, 3)

    def test_embedding_rejects_a_different_index_dimension(self):
        with self.assertRaisesRegex(ValueError, '768 chiều'):
            GeminiEmbedder(client=object(), dimensions=3072)

    def test_real_controller_client_uses_function_calling_api_version(self):
        controller = GeminiController(api_key='placeholder')
        try:
            self.assertEqual(
                controller.client._api_client._http_options.api_version,
                'v1beta',
            )
        finally:
            controller.client.close()


if __name__ == '__main__':
    unittest.main()
