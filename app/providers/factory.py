"""Construct configured provider adapters without coupling callers to them."""

from __future__ import annotations

from app.providers.base import AnswerProvider, EmbeddingProvider
from app.providers.ollama import OllamaAnswerProvider, OllamaEmbeddingProvider


def create_embedding_provider(
    provider: str,
    *,
    model: str,
    dimension: int,
    url: str,
    timeout: float,
    document_prefix: str,
    query_prefix: str,
) -> EmbeddingProvider:
    if provider == "ollama":
        return OllamaEmbeddingProvider(
            model=model,
            dimension=dimension,
            url=url,
            timeout=timeout,
            document_prefix=document_prefix,
            query_prefix=query_prefix,
        )
    raise ValueError(
        f"Unknown embedding provider {provider!r}. Implemented: 'ollama'."
    )


def create_answer_provider(
    provider: str, *, model: str, url: str, timeout: float
) -> AnswerProvider:
    if provider == "ollama":
        return OllamaAnswerProvider(model=model, url=url, timeout=timeout)
    raise ValueError(f"Unknown answer provider {provider!r}. Implemented: 'ollama'.")
