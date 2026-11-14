"""Persistent vector storage and cosine search via PostgreSQL + pgvector."""
import json
import math
from typing import Any

import psycopg
from psycopg.rows import dict_row

from .models import Chunk


class PgVectorStore:
    def __init__(self, database_url: str, dimensions: int) -> None:
        if not database_url:
            raise ValueError("DATABASE_URL is required")
        if dimensions <= 0:
            raise ValueError("dimensions must be greater than zero")
        self.database_url = database_url
        self.dimensions = dimensions
        self.table_name = f"lab_chunks_{dimensions}"

    def create_schema(self) -> None:
        with psycopg.connect(self.database_url) as connection:
            connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    strategy TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    page INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    metadata JSONB NOT NULL,
                    embedding VECTOR({self.dimensions}) NOT NULL,
                    PRIMARY KEY (strategy, chunk_id)
                )
                """
            )
            connection.execute(
                f"""
                CREATE INDEX IF NOT EXISTS {self.table_name}_strategy_page_idx
                ON {self.table_name} (strategy, document_id, page)
                """
            )

    def describe(self) -> dict[str, str]:
        with psycopg.connect(self.database_url) as connection:
            row = connection.execute(
                """
                SELECT current_setting('server_version') AS postgres_version,
                       (SELECT extversion FROM pg_extension WHERE extname = 'vector')
                           AS pgvector_version
                """
            ).fetchone()
        if row is None or row[1] is None:
            raise RuntimeError("pgvector extension is not available in the database")
        return {"postgres_version": row[0], "pgvector_version": row[1]}

    def _vector_literal(self, vector: list[float]) -> str:
        if len(vector) != self.dimensions:
            raise ValueError(
                f"Expected {self.dimensions} vector values, got {len(vector)}"
            )
        if not all(math.isfinite(value) for value in vector):
            raise ValueError("Vector values must all be finite numbers")
        return "[" + ",".join(repr(float(value)) for value in vector) + "]"

    def upsert(
        self,
        strategy: str,
        chunks: list[Chunk],
        vectors: list[list[float]],
    ) -> None:
        if not strategy:
            raise ValueError("strategy must not be empty")
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")

        rows = []
        seen_ids: set[str] = set()
        for chunk, vector in zip(chunks, vectors):
            if chunk.chunk_id in seen_ids:
                raise ValueError(f"Duplicate chunk ID: {chunk.chunk_id}")
            seen_ids.add(chunk.chunk_id)
            rows.append(
                (
                    strategy,
                    chunk.chunk_id,
                    chunk.document_id,
                    chunk.source,
                    chunk.page,
                    chunk.text,
                    json.dumps(chunk.metadata, ensure_ascii=False),
                    self._vector_literal(vector),
                )
            )

        with psycopg.connect(self.database_url) as connection:
            connection.execute(
                f"DELETE FROM {self.table_name} WHERE strategy = %s",
                (strategy,),
            )
            if rows:
                with connection.cursor() as cursor:
                    cursor.executemany(
                        f"""
                        INSERT INTO {self.table_name}
                            (strategy, chunk_id, document_id, source, page,
                             content, metadata, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::vector)
                        """,
                        rows,
                    )

    def search(
        self,
        strategy: str,
        query_vector: list[float],
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        vector_literal = self._vector_literal(query_vector)
        filter_json = json.dumps(filters or {}, ensure_ascii=False)
        with psycopg.connect(self.database_url, row_factory=dict_row) as connection:
            rows = connection.execute(
                f"""
                SELECT chunk_id, document_id, source, page, content, metadata,
                       1 - (embedding <=> %s::vector) AS score
                FROM {self.table_name}
                WHERE strategy = %s AND metadata @> %s::jsonb
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (vector_literal, strategy, filter_json, vector_literal, top_k),
            ).fetchall()
        return [
            {
                "chunk_id": row["chunk_id"],
                "document_id": row["document_id"],
                "source": row["source"],
                "page": row["page"],
                "text": row["content"],
                "metadata": row["metadata"],
                "score": float(row["score"]),
            }
            for row in rows
        ]
