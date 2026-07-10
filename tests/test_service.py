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


@pytest.mark.parametrize("mode", ["advisor", "strict"])
def test_ask_forwards_mode_and_top_k(monkeypatch, mode):
    rag = service.RAGService()
    rows = [(0.1, 1, "Product", "https://x/p/1", None, '["Aqua"]')]
    captured = {}
    monkeypatch.setattr(rag, "_connect", lambda **_kwargs: FakeConnection())
    monkeypatch.setattr(service, "get_embedding", lambda *_args, **_kwargs: [1.0])

    def fake_search(_conn, _vector, top_k):
        captured["top_k"] = top_k
        return rows

    def fake_answer(question, context, **kwargs):
        captured.update(question=question, context=context, **kwargs)
        return "answer"

    monkeypatch.setattr(service, "search", fake_search)
    monkeypatch.setattr(service, "generate_answer", fake_answer)
    result = rag.ask("  useful question  ", mode=mode, top_k=4)
    assert result.question == "useful question"
    assert result.mode == mode
    assert captured["mode"] == mode
    assert captured["top_k"] == 4


def test_no_results_skips_answer_model(monkeypatch):
    rag = service.RAGService()
    monkeypatch.setattr(rag, "_connect", lambda **_kwargs: FakeConnection())
    monkeypatch.setattr(service, "get_embedding", lambda *_args, **_kwargs: [1.0])
    monkeypatch.setattr(service, "search", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        service,
        "generate_answer",
        lambda *_args, **_kwargs: pytest.fail("answer model should not be called"),
    )
    result = rag.ask("question")
    assert result.products == []
    assert "No matching products" in result.answer


def test_stream_ask_yields_progress_evidence_tokens_and_done(monkeypatch):
    rag = service.RAGService()
    rows = [(0.1, 1, "Product", "https://x/p/1", None, '["Aqua"]')]
    monkeypatch.setattr(rag, "_connect", lambda **_kwargs: FakeConnection())
    monkeypatch.setattr(service, "get_embedding", lambda *_args, **_kwargs: [1.0])
    monkeypatch.setattr(service, "search", lambda *_args, **_kwargs: rows)
    monkeypatch.setattr(
        service,
        "_stream_answer_ollama",
        lambda *_args, **_kwargs: iter(["Hello ", "**world**"]),
    )
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
    rag = service.RAGService()
    monkeypatch.setattr(rag, "_connect", lambda **_kwargs: FakeConnection())
    monkeypatch.setattr(service, "get_embedding", lambda *_args, **_kwargs: [1.0])
    monkeypatch.setattr(service, "search", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        service,
        "_stream_answer_ollama",
        lambda *_args, **_kwargs: pytest.fail("answer model should not be called"),
    )
    events = list(rag.stream_ask("question"))
    assert any(event["type"] == "evidence" and event["products"] == [] for event in events)
    assert events[-1]["type"] == "done"


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
