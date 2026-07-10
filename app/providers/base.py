"""Provider-neutral contracts used by indexing and question answering."""

from __future__ import annotations

from typing import Iterator, Protocol, runtime_checkable


class ProviderError(Exception):
    """A normalized provider failure safe to translate at the service edge."""

    def __init__(self, code: str, message: str, remediation: str):
        super().__init__(message)
        self.code = code
        self.message = message
        self.remediation = remediation


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Turn document and query text into vectors in one compatible space."""

    provider: str
    model: str
    dimension: int
    document_prefix: str
    query_prefix: str

    def embed_query(self, text: str) -> list[float]: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


@runtime_checkable
class AnswerProvider(Protocol):
    """Generate complete or streamed text from a system/user prompt pair."""

    provider: str
    model: str

    def generate(self, system: str, prompt: str) -> str: ...

    def stream(self, system: str, prompt: str) -> Iterator[str]: ...
