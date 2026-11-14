import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch


class GeminiEmbedderTests(unittest.TestCase):
    def test_embedding_2_respects_estimated_token_window(self) -> None:
        from src.embeddings import MyEmbedder

        response = SimpleNamespace(
            embeddings=[SimpleNamespace(values=[0.1, 0.2, 0.3])]
        )
        clock = [0.0]

        def advance_clock(seconds: float) -> None:
            clock[0] += seconds

        with patch.dict(
            os.environ,
            {
                "GEMINI_API_KEY": "test-key",
                "EMBEDDING_DIMENSIONS": "3",
                "GEMINI_EMBEDDING_TPM_LIMIT": "20",
            },
            clear=False,
        ), patch("src.embeddings.load_dotenv"), patch(
            "google.genai.Client"
        ) as client_factory, patch(
            "src.embeddings.time.monotonic", side_effect=lambda: clock[0]
        ), patch(
            "src.embeddings.time.sleep", side_effect=advance_clock
        ):
            client_factory.return_value.models.embed_content.return_value = response
            embedder = MyEmbedder("gemini-embedding-2")

            embedder.embed_documents(["x" * 24, "y" * 24])

        self.assertGreaterEqual(clock[0], 60.0)
        self.assertEqual(client_factory.return_value.models.embed_content.call_count, 2)

    def test_embedding_2_embeds_each_document_and_formats_query(self) -> None:
        from src.embeddings import MyEmbedder

        response = SimpleNamespace(
            embeddings=[SimpleNamespace(values=[0.1, 0.2, 0.3])]
        )
        with patch.dict(
            os.environ,
            {"GEMINI_API_KEY": "test-key", "EMBEDDING_DIMENSIONS": "3"},
            clear=False,
        ), patch("src.embeddings.load_dotenv"), patch(
            "google.genai.Client"
        ) as client_factory:
            client = client_factory.return_value
            client.models.embed_content.return_value = response
            embedder = MyEmbedder("gemini-embedding-2")

            documents = embedder.embed_documents(["first", "second"])
            query = embedder.embed_query("find evidence")

        self.assertEqual(documents, [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]])
        self.assertEqual(query, [0.1, 0.2, 0.3])
        self.assertEqual(client.models.embed_content.call_count, 3)
        query_call = client.models.embed_content.call_args
        self.assertEqual(
            query_call.kwargs["contents"],
            "task: search result | query: find evidence",
        )
        client_factory.assert_called_once_with(api_key="test-key")

    def test_requires_api_key(self) -> None:
        from src.embeddings import MyEmbedder

        with patch.dict(os.environ, {}, clear=True), patch(
            "src.embeddings.load_dotenv"
        ):
            with self.assertRaisesRegex(RuntimeError, "GEMINI_API_KEY"):
                MyEmbedder("gemini-embedding-2")

    def test_embedding_001_uses_retrieval_task_types(self) -> None:
        from src.embeddings import MyEmbedder

        document_response = SimpleNamespace(
            embeddings=[
                SimpleNamespace(values=[0.1, 0.2, 0.3]),
                SimpleNamespace(values=[0.4, 0.5, 0.6]),
            ]
        )
        query_response = SimpleNamespace(
            embeddings=[SimpleNamespace(values=[0.1, 0.2, 0.3])]
        )
        with patch.dict(
            os.environ,
            {"GEMINI_API_KEY": "test-key", "EMBEDDING_DIMENSIONS": "3"},
            clear=False,
        ), patch("src.embeddings.load_dotenv"), patch(
            "google.genai.Client"
        ) as client_factory:
            client = client_factory.return_value
            client.models.embed_content.side_effect = [
                document_response,
                query_response,
            ]
            embedder = MyEmbedder("gemini-embedding-001")

            documents = embedder.embed_documents(["first", "second"])
            query = embedder.embed_query("find evidence")

        self.assertEqual(len(documents), 2)
        self.assertEqual(query, [0.1, 0.2, 0.3])
        document_call = client.models.embed_content.call_args_list[0]
        query_call = client.models.embed_content.call_args_list[1]
        self.assertEqual(
            document_call.kwargs["config"].task_type,
            "RETRIEVAL_DOCUMENT",
        )
        self.assertEqual(
            query_call.kwargs["config"].task_type,
            "RETRIEVAL_QUERY",
        )


if __name__ == "__main__":
    unittest.main()
