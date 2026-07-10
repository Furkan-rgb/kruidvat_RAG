"""Ollama adapters for the provider-neutral model contracts."""

from __future__ import annotations

import json
import socket
from collections.abc import Callable, Iterator
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

from app.providers.base import ProviderError

PostJson = Callable[[str, dict[str, Any], float], dict[str, Any]]


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    req = urllib_request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib_request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", errors="ignore")
    return json.loads(body)


def translate_error(exc: Exception, model: str) -> ProviderError:
    """Normalize Ollama transport, timeout, and missing-model failures."""
    if isinstance(exc, urllib_error.HTTPError):
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="ignore")
        except Exception:
            pass
        if exc.code == 404 or "not found" in body.lower():
            return ProviderError(
                "ollama_model_missing",
                f"The configured Ollama model {model!r} is not available.",
                f"Run `ollama pull {model}` and try again.",
            )
    message = str(exc).lower()
    if "model" in message and "not found" in message:
        return ProviderError(
            "ollama_model_missing",
            f"The configured Ollama model {model!r} is not available.",
            f"Run `ollama pull {model}` and try again.",
        )
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return ProviderError(
            "ollama_timeout",
            "Ollama did not respond before the request timed out.",
            "Check that Ollama is responsive, or increase OLLAMA_TIMEOUT in config.py.",
        )
    if isinstance(exc, urllib_error.URLError) and isinstance(
        exc.reason, (TimeoutError, socket.timeout)
    ):
        return ProviderError(
            "ollama_timeout",
            "Ollama did not respond before the request timed out.",
            "Check that Ollama is responsive, or increase OLLAMA_TIMEOUT in config.py.",
        )
    return ProviderError(
        "ollama_unavailable",
        "The configured Ollama service is unavailable.",
        "Start Ollama and confirm the configured host in config.py is reachable.",
    )


class OllamaEmbeddingProvider:
    provider = "ollama"

    def __init__(
        self,
        *,
        model: str,
        dimension: int,
        url: str,
        timeout: float,
        document_prefix: str = "",
        query_prefix: str = "",
        _post: PostJson | None = None,
    ):
        self.model = model
        self.dimension = dimension
        self.url = url
        self.timeout = timeout
        self.document_prefix = document_prefix
        self.query_prefix = query_prefix
        self._post = _post or post_json

    def _embed(self, text: str) -> list[float]:
        try:
            vector = self._post(
                self.url, {"model": self.model, "prompt": text}, self.timeout
            )["embedding"]
            normalized = [float(value) for value in vector]
            if self.dimension > 0 and len(normalized) != self.dimension:
                raise ProviderError(
                    "embedding_dimension_mismatch",
                    f"Embedding model {self.model!r} returned {len(normalized)} values; "
                    f"the configured dimension is {self.dimension}.",
                    "Correct EMBED_DIM and rebuild the index with `python embed.py --reset`.",
                )
            return normalized
        except ProviderError:
            raise
        except Exception as exc:
            raise translate_error(exc, self.model) from exc

    def embed_query(self, text: str) -> list[float]:
        return self._embed(self.query_prefix + text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(self.document_prefix + text) for text in texts]


class OllamaAnswerProvider:
    provider = "ollama"

    def __init__(
        self,
        *,
        model: str,
        url: str,
        timeout: float,
        _post: PostJson | None = None,
    ):
        self.model = model
        self.url = url
        self.timeout = timeout
        self._post = _post or post_json

    def _payload(self, system: str, prompt: str, *, stream: bool) -> dict[str, Any]:
        return {
            "model": self.model,
            "stream": stream,
            "think": False,
            "system": system,
            "options": {"temperature": 0},
            "prompt": prompt,
        }

    def generate(self, system: str, prompt: str) -> str:
        try:
            return self._post(
                self.url, self._payload(system, prompt, stream=False), self.timeout
            ).get("response", "").strip()
        except ProviderError:
            raise
        except Exception as exc:
            raise translate_error(exc, self.model) from exc

    def stream(self, system: str, prompt: str) -> Iterator[str]:
        payload = self._payload(system, prompt, stream=True)
        req = urllib_request.Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib_request.urlopen(req, timeout=self.timeout) as resp:
                for raw_line in resp:
                    if not raw_line.strip():
                        continue
                    body = json.loads(raw_line.decode("utf-8", errors="ignore"))
                    if body.get("error"):
                        raise RuntimeError(str(body["error"]))
                    chunk = body.get("response", "")
                    if chunk:
                        yield chunk
        except ProviderError:
            raise
        except Exception as exc:
            raise translate_error(exc, self.model) from exc
