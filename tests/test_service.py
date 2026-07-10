"""Tests for the shared RAG application service (Ollama is always mocked)."""

import json
import sqlite3
from urllib import error as urllib_error

import pytest

from app import service
from lib import db


def test_validation_rejects_blank_question_and_bad_limits():
    with pytest.raises(service.InputError):
        service.RAGService.validate("   ", "advisor", 10)
    with pytest.raises(service.InputError):
        service.RAGService.validate("hello", "other", 10)
    with pytest.raises(service.InputError):
        service.RAGService.validate("hello", "advisor", 26)


def test_parse_ingredients_normalizes_json_array():
    assert service.parse_ingredients('[" Aqua ", "Glycerin", ""]') == ["Aqua", "Glycerin"]
    assert service.parse_ingredients("not-json") == []
    assert service.parse_ingredients('{"ingredient": "Aqua"}') == []


def test_rows_to_products_preserves_rank_distance_and_fields():
    rows = [(0.125, 7, "Mousse", "https://x/p/7", "Light hold", '["Aqua"]')]
    product = service.rows_to_products(rows)[0]
    assert product.rank == 1
    assert product.distance == 0.125
    assert product.id == 7
    assert product.description == "Light hold"
    assert product.ingredients == ["Aqua"]


class FakeConnection:
    def close(self):
        pass


class FakeEmbeddingProvider:
    provider = "fake-embeddings"
    model = "fake-embed-model"
    dimension = 1
    document_prefix = "document: "
    query_prefix = "query: "

    def embed_query(self, text):
        self.query = text
        return [1.0]

    def embed_documents(self, texts):
        return [[1.0] for _text in texts]


class FakeAnswerProvider:
    provider = "fake-answers"
    model = "fake-answer-model"

    def __init__(self, answer="answer", chunks=None):
        self.answer = answer
        self.chunks = chunks or [answer]
        self.calls = []

    def generate(self, system, prompt):
        self.calls.append((system, prompt))
        return self.answer

    def stream(self, system, prompt):
        self.calls.append((system, prompt))
        yield from self.chunks


def fake_rag(*, answer="answer", chunks=None):
    answer_provider = FakeAnswerProvider(answer, chunks)
    rag = service.RAGService(
        embedding_provider=FakeEmbeddingProvider(),
        answer_provider=answer_provider,
    )
    return rag, answer_provider


@pytest.mark.parametrize("mode", ["advisor", "strict"])
def test_ask_forwards_mode_and_top_k(monkeypatch, mode):
    rag, answer_provider = fake_rag()
    rows = [(0.1, 1, "Product", "https://x/p/1", None, '["Aqua"]')]
    captured = {}
    monkeypatch.setattr(rag, "_connect", lambda **_kwargs: FakeConnection())

    def fake_search(_conn, _vector, top_k):
        captured["top_k"] = top_k
        return rows

    monkeypatch.setattr(service, "search", fake_search)
    result = rag.ask("  useful question  ", mode=mode, top_k=4)
    assert result.question == "useful question"
    assert result.mode == mode
    assert captured["top_k"] == 4
    assert answer_provider.calls[0][0] == service.SYSTEM_PROMPTS[mode]
    assert "Question: useful question" in answer_provider.calls[0][1]


def test_no_results_skips_answer_model(monkeypatch):
    rag, answer_provider = fake_rag()
    monkeypatch.setattr(rag, "_connect", lambda **_kwargs: FakeConnection())
    monkeypatch.setattr(service, "search", lambda *_args, **_kwargs: [])
    result = rag.ask("question")
    assert result.products == []
    assert "No matching products" in result.answer
    assert answer_provider.calls == []


def test_stream_ask_yields_progress_evidence_tokens_and_done(monkeypatch):
    rag, _answer_provider = fake_rag(chunks=["Hello ", "**world**"])
    rows = [(0.1, 1, "Product", "https://x/p/1", None, '["Aqua"]')]
    monkeypatch.setattr(rag, "_connect", lambda **_kwargs: FakeConnection())
    monkeypatch.setattr(service, "search", lambda *_args, **_kwargs: rows)
    events = list(rag.stream_ask("question", mode="strict", top_k=3))
    assert [event["stage"] for event in events if event["type"] == "status"] == [
        "embedding",
        "retrieving",
        "generating",
    ]
    evidence = next(event for event in events if event["type"] == "evidence")
    assert evidence["products"][0]["distance"] == 0.1
    assert "".join(event["text"] for event in events if event["type"] == "token") == "Hello **world**"
    assert events[-1]["type"] == "done"
    assert events[-1]["answer"] == "Hello **world**"


def test_stream_no_results_finishes_without_answer_model(monkeypatch):
    rag, answer_provider = fake_rag()
    monkeypatch.setattr(rag, "_connect", lambda **_kwargs: FakeConnection())
    monkeypatch.setattr(service, "search", lambda *_args, **_kwargs: [])
    events = list(rag.stream_ask("question"))
    assert any(event["type"] == "evidence" and event["products"] == [] for event in events)
    assert events[-1]["type"] == "done"
    assert answer_provider.calls == []


def test_missing_database_does_not_create_it(tmp_path):
    path = tmp_path / "missing.db"
    rag = service.RAGService(db_path=str(path))
    with pytest.raises(service.ServiceError, match="does not exist") as caught:
        rag._connect(require_vector=False)
    assert caught.value.code == "database_missing"
    assert not path.exists()


def test_missing_vector_index_is_translated(tmp_path):
    path = tmp_path / "products.db"
    conn = db.setup_db(str(path))
    conn.close()
    with pytest.raises(service.ServiceError) as caught:
        service.RAGService(db_path=str(path))._connect(require_vector=True)
    assert caught.value.code == "vector_index_missing"


def test_sqlite_vec_load_failure_is_translated(tmp_path, monkeypatch):
    path = tmp_path / "products.db"
    conn = db.setup_db(str(path))
    conn.execute("CREATE TABLE vec_products(product_id INTEGER)")
    conn.commit()
    conn.close()
    monkeypatch.setattr(
        service.sqlite_vec,
        "load",
        lambda _conn: (_ for _ in ()).throw(RuntimeError("no extension")),
    )
    with pytest.raises(service.ServiceError) as caught:
        service.RAGService(db_path=str(path))._connect(require_vector=True)
    assert caught.value.code == "sqlite_vec_unavailable"


def test_missing_products_table_is_reported_in_health(tmp_path):
    path = tmp_path / "empty.db"
    sqlite3.connect(path).close()
    health = service.RAGService(db_path=str(path)).health()
    assert health["status"] == "products_table_missing"
    assert health["products_table_exists"] is False


def test_ollama_connection_and_model_errors_are_translated():
    unavailable = service._translate_ollama_error(
        urllib_error.URLError("connection refused"), "embeddinggemma"
    )
    assert unavailable.code == "ollama_unavailable"
    missing = service._translate_ollama_error(
        urllib_error.HTTPError("http://x", 404, "not found", {}, None), "missing-model"
    )
    assert missing.code == "ollama_model_missing"


def test_provider_error_is_translated_without_provider_specific_service_logic(monkeypatch):
    class FailingEmbeddingProvider(FakeEmbeddingProvider):
        def embed_query(self, _text):
            raise service.ProviderError("cloud_timeout", "Too slow.", "Try again.")

    rag = service.RAGService(
        embedding_provider=FailingEmbeddingProvider(),
        answer_provider=FakeAnswerProvider(),
    )
    monkeypatch.setattr(rag, "_connect", lambda **_kwargs: FakeConnection())
    with pytest.raises(service.ServiceError) as caught:
        rag.retrieve("question")
    assert caught.value.code == "cloud_timeout"
    assert caught.value.remediation == "Try again."


def test_get_product_returns_parsed_record(tmp_path):
    path = tmp_path / "products.db"
    conn = db.setup_db(str(path))
    db.save_products_batch(
        conn,
        [("Mousse", "https://x/p/1", "Light", json.dumps(["Aqua"]), "2026-01-01")],
    )
    product_id = conn.execute("SELECT id FROM products").fetchone()[0]
    conn.close()
    product = service.RAGService(db_path=str(path)).get_product(product_id)
    assert product is not None
    assert product.ingredients == ["Aqua"]
    assert product.scraped_at == "2026-01-01"
