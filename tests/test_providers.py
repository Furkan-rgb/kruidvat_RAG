"""Provider contract and Ollama-adapter tests without live model calls."""

import json

import pytest

from app.providers import ollama
from app.providers import (
    OllamaAnswerProvider,
    OllamaEmbeddingProvider,
    ProviderError,
    create_answer_provider,
    create_embedding_provider,
)


def test_ollama_embedding_adapter_applies_distinct_prefixes():
    prompts = []

    def post(_url, payload, _timeout):
        prompts.append(payload["prompt"])
        return {"embedding": [1, 2]}

    provider = OllamaEmbeddingProvider(
        model="embed",
        dimension=2,
        url="http://ollama/embeddings",
        timeout=1,
        document_prefix="doc: ",
        query_prefix="query: ",
        _post=post,
    )
    assert provider.embed_query("hello") == [1.0, 2.0]
    assert provider.embed_documents(["one", "two"]) == [[1.0, 2.0], [1.0, 2.0]]
    assert prompts == ["query: hello", "doc: one", "doc: two"]


def test_ollama_embedding_adapter_rejects_wrong_dimension():
    provider = OllamaEmbeddingProvider(
        model="embed",
        dimension=3,
        url="u",
        timeout=1,
        _post=lambda *_args: {"embedding": [1, 2]},
    )
    with pytest.raises(ProviderError) as caught:
        provider.embed_query("hello")
    assert caught.value.code == "embedding_dimension_mismatch"


def test_ollama_answer_adapter_preserves_generation_payload():
    captured = {}

    def post(url, payload, timeout):
        captured.update(url=url, payload=payload, timeout=timeout)
        return {"response": "  answer  "}

    provider = OllamaAnswerProvider(model="answer", url="u", timeout=5, _post=post)
    assert provider.generate("system", "prompt") == "answer"
    assert captured["payload"] == {
        "model": "answer",
        "stream": False,
        "think": False,
        "system": "system",
        "options": {"temperature": 0},
        "prompt": "prompt",
    }


def test_ollama_answer_adapter_streams_ndjson(monkeypatch):
    captured = {}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def __iter__(self):
            return iter(
                [
                    b'{"response":"Hello ","done":false}\n',
                    b'{"response":"world","done":false}\n',
                    b'{"done":true}\n',
                ]
            )

    def urlopen(request, timeout):
        captured["payload"] = json.loads(request.data)
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr(ollama.urllib_request, "urlopen", urlopen)
    provider = OllamaAnswerProvider(model="answer", url="http://ollama/generate", timeout=5)
    assert list(provider.stream("system", "prompt")) == ["Hello ", "world"]
    assert captured["payload"]["stream"] is True
    assert captured["payload"]["think"] is False
    assert captured["timeout"] == 5


def test_factories_reject_unknown_providers():
    with pytest.raises(ValueError, match="embedding provider"):
        create_embedding_provider(
            "unknown",
            model="m",
            dimension=1,
            url="u",
            timeout=1,
            document_prefix="",
            query_prefix="",
        )
    with pytest.raises(ValueError, match="answer provider"):
        create_answer_provider("unknown", model="m", url="u", timeout=1)
