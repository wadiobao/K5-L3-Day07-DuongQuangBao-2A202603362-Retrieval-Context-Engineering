import unittest
from unittest.mock import patch

from src.store import PgVectorStore


class PgVectorStoreTests(unittest.TestCase):
    def test_vector_literal_validates_dimension_and_values(self) -> None:
        store = PgVectorStore("postgresql://unused", 2)

        self.assertEqual(store._vector_literal([0.25, -1.0]), "[0.25,-1.0]")
        with self.assertRaisesRegex(ValueError, "Expected 2"):
            store._vector_literal([0.25])
        with self.assertRaisesRegex(ValueError, "finite"):
            store._vector_literal([float("nan"), 1.0])

    def test_search_parameterizes_metadata_filters(self) -> None:
        store = PgVectorStore("postgresql://unused", 2)
        with patch("src.store.psycopg.connect") as connect:
            fake_connection = connect.return_value.__enter__.return_value
            fake_connection.execute.return_value.fetchall.return_value = [
                {
                    "chunk_id": "c1",
                    "document_id": "book",
                    "source": "source.txt",
                    "page": 3,
                    "content": "evidence",
                    "metadata": {"section_id": "chapter-i"},
                    "score": 0.9,
                }
            ]
            results = store.search(
                "C",
                [0.1, 0.2],
                filters={"section_id": "chapter-i'; DROP TABLE lab_chunks_2; --"},
            )
            query = fake_connection.execute.call_args.args[0]
            parameters = fake_connection.execute.call_args.args[1]

        self.assertEqual(results[0]["text"], "evidence")
        self.assertNotIn("DROP TABLE", query)
        self.assertIn("DROP TABLE", parameters[2])


if __name__ == "__main__":
    unittest.main()
