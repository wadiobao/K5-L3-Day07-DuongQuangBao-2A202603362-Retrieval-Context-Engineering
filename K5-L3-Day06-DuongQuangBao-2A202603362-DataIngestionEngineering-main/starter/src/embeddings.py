"""Gemini embedding provider for the lab's Embedder contract."""
import logging
import os
import time
from collections import deque
from typing import Protocol

from dotenv import load_dotenv

logger = logging.getLogger(__name__)
_TOKEN_WINDOW_SECONDS = 60.0
_ESTIMATED_CHARS_PER_TOKEN = 3


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        ...


class MyEmbedder:
    """Create real Gemini embeddings; never silently fall back to mock vectors."""

    def __init__(
        self,
        model_name: str | None = None,
        dimensions: int | None = None,
    ) -> None:
        load_dotenv()
        self.model_name = (
            model_name
            or os.getenv("EMBEDDING_MODEL")
            or "gemini-embedding-2"
        )
        if self.model_name.startswith("models/"):
            self.model_name = self.model_name.removeprefix("models/")
        if self.model_name not in {"gemini-embedding-2", "gemini-embedding-001"}:
            raise ValueError(
                "EMBEDDING_MODEL must be gemini-embedding-2 or "
                "gemini-embedding-001"
            )

        try:
            self.dimensions = (
                dimensions
                if dimensions is not None
                else int(os.getenv("EMBEDDING_DIMENSIONS", "768"))
            )
        except ValueError as exc:
            raise ValueError("EMBEDDING_DIMENSIONS must be an integer") from exc
        if self.dimensions <= 0:
            raise ValueError("EMBEDDING_DIMENSIONS must be greater than zero")

        try:
            self.tokens_per_minute = int(
                os.getenv("GEMINI_EMBEDDING_TPM_LIMIT", "20000")
            )
        except ValueError as exc:
            raise ValueError("GEMINI_EMBEDDING_TPM_LIMIT must be an integer") from exc
        if self.tokens_per_minute <= 0:
            raise ValueError("GEMINI_EMBEDDING_TPM_LIMIT must be greater than zero")

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")

        from google import genai

        self._client = genai.Client(api_key=api_key)
        self.request_count = 0
        self.input_token_count: int | None = None
        self._recent_tokens: deque[tuple[float, int]] = deque()
        self._recent_token_total = 0

    def _estimate_tokens(self, contents: object) -> int:
        if isinstance(contents, str):
            texts = [contents]
        elif isinstance(contents, (list, tuple)) and all(
            isinstance(text, str) for text in contents
        ):
            texts = list(contents)
        else:
            raise TypeError("Gemini embedding contents must be text or a list of text")
        return max(
            1,
            sum(
                (len(text.encode("utf-8")) + _ESTIMATED_CHARS_PER_TOKEN - 1)
                // _ESTIMATED_CHARS_PER_TOKEN
                for text in texts
            ),
        )

    def _reserve_token_capacity(self, estimated_tokens: int) -> None:
        if estimated_tokens > self.tokens_per_minute:
            raise ValueError(
                f"One Gemini embedding request is estimated at {estimated_tokens} "
                f"tokens, above GEMINI_EMBEDDING_TPM_LIMIT={self.tokens_per_minute}; "
                "raise the limit or reduce the batch size"
            )

        while True:
            now = time.monotonic()
            while (
                self._recent_tokens
                and now - self._recent_tokens[0][0] >= _TOKEN_WINDOW_SECONDS
            ):
                _, expired_tokens = self._recent_tokens.popleft()
                self._recent_token_total -= expired_tokens

            if self._recent_token_total + estimated_tokens <= self.tokens_per_minute:
                self._recent_tokens.append((now, estimated_tokens))
                self._recent_token_total += estimated_tokens
                return

            wait_seconds = max(
                0.01,
                self._recent_tokens[0][0] + _TOKEN_WINDOW_SECONDS - now,
            )
            logger.info(
                "Waiting %.1fs to stay within the configured Gemini embedding "
                "token rate limit",
                wait_seconds,
            )
            time.sleep(wait_seconds)

    def _embed_content(self, **kwargs: object) -> object:
        self._reserve_token_capacity(self._estimate_tokens(kwargs["contents"]))
        result = self._client.models.embed_content(**kwargs)
        self.request_count += 1
        usage = getattr(result, "usage_metadata", None)
        token_count = getattr(usage, "total_token_count", None)
        if token_count is not None:
            self.input_token_count = (self.input_token_count or 0) + int(token_count)
        return result

    def _values(self, result: object) -> list[list[float]]:
        embeddings = getattr(result, "embeddings", None)
        if embeddings is None:
            raise RuntimeError("Gemini returned no embeddings")

        vectors: list[list[float]] = []
        for embedding in embeddings:
            values = getattr(embedding, "values", None)
            if values is None:
                raise RuntimeError("Gemini returned an embedding without values")
            vector = [float(value) for value in values]
            if len(vector) != self.dimensions:
                raise RuntimeError(
                    f"Gemini returned {len(vector)} dimensions; "
                    f"expected {self.dimensions}"
                )
            vectors.append(vector)
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        from google.genai import types

        if self.model_name == "gemini-embedding-001":
            vectors = []
            batch: list[str] = []
            batch_tokens = 0
            for text in texts:
                text_tokens = self._estimate_tokens(text)
                if text_tokens > self.tokens_per_minute:
                    raise ValueError(
                        f"One document is estimated at {text_tokens} tokens, above "
                        "GEMINI_EMBEDDING_TPM_LIMIT"
                    )
                if batch and (
                    len(batch) >= 64
                    or batch_tokens + text_tokens > self.tokens_per_minute
                ):
                    result = self._embed_content(
                        model=self.model_name,
                        contents=batch,
                        config=types.EmbedContentConfig(
                            task_type="RETRIEVAL_DOCUMENT",
                            output_dimensionality=self.dimensions,
                        ),
                    )
                    vectors.extend(self._values(result))
                    batch = []
                    batch_tokens = 0
                batch.append(text)
                batch_tokens += text_tokens
            if batch:
                result = self._embed_content(
                    model=self.model_name,
                    contents=batch,
                    config=types.EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT",
                        output_dimensionality=self.dimensions,
                    ),
                )
                vectors.extend(self._values(result))
        else:
            vectors = []
            for text in texts:
                result = self._embed_content(
                    model=self.model_name,
                    contents=f"title: none | text: {text}",
                    config=types.EmbedContentConfig(
                        output_dimensionality=self.dimensions,
                    ),
                )
                vectors.extend(self._values(result))

        if len(vectors) != len(texts):
            raise RuntimeError(
                f"Gemini returned {len(vectors)} embeddings for {len(texts)} texts"
            )
        return vectors

    def embed_query(self, question: str) -> list[float]:
        from google.genai import types

        if self.model_name == "gemini-embedding-001":
            contents = question
            config = types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=self.dimensions,
            )
        else:
            contents = f"task: search result | query: {question}"
            config = types.EmbedContentConfig(
                output_dimensionality=self.dimensions,
            )

        vectors = self._values(
            self._embed_content(
                model=self.model_name,
                contents=contents,
                config=config,
            )
        )
        if len(vectors) != 1:
            raise RuntimeError(
                f"Gemini returned {len(vectors)} embeddings for one query"
            )
        return vectors[0]
