"""Dense, lexical, and reciprocal-rank-fusion retrievers for Day 7."""
import json
import math
from typing import Any, Protocol

from baseline import TfidfIndex


class Embedder(Protocol):
    dimensions: int

    def embed_query(self, query: str) -> list[float]: ...


class DenseRetriever:
    """Query the Day 6 pgvector table without rebuilding its index."""

    def __init__(
        self,
        database_url: str,
        embedder: Embedder,
        *,
        strategy: str = 'C',
    ) -> None:
        if not database_url.strip():
            raise ValueError('DATABASE_URL không được để trống')
        if not strategy.strip():
            raise ValueError('strategy không được để trống')
        if type(embedder.dimensions) is not int or embedder.dimensions <= 0:
            raise ValueError('embedding dimensions phải là số nguyên dương')
        self.database_url = database_url
        self.embedder = embedder
        self.strategy = strategy
        self.table_name = f'lab_chunks_{embedder.dimensions}'

    def search(self, query: str, *, filters: dict[str, Any] | None, top_k: int) -> list[dict[str, Any]]:
        if not query.strip() or top_k <= 0:
            raise ValueError('query phải khác rỗng và top_k phải lớn hơn 0')
        vector = self.embedder.embed_query(query)
        if len(vector) != self.embedder.dimensions or not all(math.isfinite(v) for v in vector):
            raise ValueError('query embedding không hợp lệ hoặc sai số chiều')
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:
            raise RuntimeError('Thiếu psycopg; cài starter/requirements.txt') from exc

        vector_literal = '[' + ','.join(repr(float(value)) for value in vector) + ']'
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
                (vector_literal, self.strategy, filter_json, vector_literal, top_k),
            ).fetchall()
        return [
            {
                'id': row['chunk_id'],
                'source_id': row['document_id'],
                'source': row['source'],
                'page': row['page'],
                'text': row['content'],
                'metadata': row['metadata'],
                'score': float(row['score']),
                'strategy': 'dense',
            }
            for row in rows
        ]


class LexicalRetriever:
    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.index = TfidfIndex(chunks)

    def search(self, query: str, *, filters: dict[str, Any] | None, top_k: int) -> list[dict[str, Any]]:
        return self.index.search(query, top_k=top_k, filters=filters)


class HybridRetriever:
    """Fuse dense and lexical rankings with reciprocal rank fusion."""

    def __init__(
        self,
        dense: DenseRetriever,
        lexical: LexicalRetriever,
        *,
        candidate_k: int = 20,
        rrf_constant: int = 60,
    ) -> None:
        if candidate_k <= 0 or rrf_constant <= 0:
            raise ValueError('candidate_k và rrf_constant phải lớn hơn 0')
        self.dense = dense
        self.lexical = lexical
        self.candidate_k = candidate_k
        self.rrf_constant = rrf_constant

    def search(self, query: str, *, filters: dict[str, Any] | None, top_k: int) -> list[dict[str, Any]]:
        if top_k <= 0:
            raise ValueError('top_k phải lớn hơn 0')
        dense_rows = self.dense.search(query, filters=filters, top_k=self.candidate_k)
        lexical_rows = self.lexical.search(query, filters=filters, top_k=self.candidate_k)
        fused: dict[str, dict[str, Any]] = {}
        scores: dict[str, float] = {}
        for rows in (dense_rows, lexical_rows):
            for rank, row in enumerate(rows, start=1):
                chunk_id = row['id']
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (self.rrf_constant + rank)
                if chunk_id not in fused or row.get('score', 0) > fused[chunk_id].get('score', 0):
                    fused[chunk_id] = dict(row)
        ranked = []
        for chunk_id, row in fused.items():
            row['score'] = scores[chunk_id]
            row['strategy'] = 'hybrid_rrf'
            ranked.append(row)
        return sorted(ranked, key=lambda row: (-row['score'], row['id']))[:top_k]
