"""HTTP contract tests for the FastAPI application."""

import json
from dataclasses import dataclass

from fastapi.testclient import TestClient

from app import main
from app.service import ProductResult, QueryResult, ServiceError, StoredProduct


client = TestClient(main.app)


@dataclass
class FakeService:
    health_status: str = "ready"

    def health(self):
        return {
            "status": self.health_status,
            "database_exists": self.health_status != "database_missing",
            "products_table_exists": self.health_status not in {"database_missing", "products_table_missing"},
            "product_count": 2,
            "vec_products_table_exists": self.health_status == "ready",
            "embedded_product_count": 2 if self.health_status == "ready" else None,
            "models": {"embedding": "embed", "answer": "answer", "provider": "ollama"},
        }

    def ask(self, question, *, mode, top_k):
        return QueryResult(
            question=question,
            mode=mode,
            top_k=top_k,
            answer="Use the mousse.",
            products=[ProductResult(1, 0.125, 1, "Mousse", "https://x/p/1", "Light", ["Aqua"])],
            models={"embedding": "embed", "answer": "answer", "provider": "ollama"},
            elapsed_ms=12.5,
        )

    def get_product(self, product_id):
        if product_id != 1:
            return None
        return StoredProduct(1, "Mousse", "https://x/p/1", "Light", ["Aqua"], "2026-01-01")

    def stream_ask(self, question, *, mode, top_k):
        result = self.ask(question, mode=mode, top_k=top_k)
        yield {"type": "status", "stage": "embedding", "message": "Embedding..."}
        yield {
            "type": "evidence",
            "question": question,
            "mode": mode,
            "top_k": top_k,
            "products": [product.__dict__ for product in result.products],
            "models": result.models,
        }
        yield {"type": "token", "text": "Use "}
        yield {"type": "token", "text": "the mousse."}
        yield {"type": "done", "answer": result.answer, "elapsed_ms": result.elapsed_ms}


def use_service(monkeypatch, fake=None):
    fake = fake or FakeService()
    monkeypatch.setattr(main, "get_service", lambda: fake)
    return fake


def test_health(monkeypatch):
    use_service(monkeypatch)
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert response.json()["embedded_product_count"] == 2


def test_health_reports_missing_database(monkeypatch):
    use_service(monkeypatch, FakeService("database_missing"))
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["database_exists"] is False


def test_successful_ask(monkeypatch):
    use_service(monkeypatch)
    response = client.post(
        "/api/ask", json={"question": "A mousse", "mode": "strict", "top_k": 3}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "strict"
    assert body["top_k"] == 3
    assert body["products"][0]["distance"] == 0.125
    assert body["products"][0]["ingredients"] == ["Aqua"]


def test_successful_stream_has_ordered_progress_and_answer_events(monkeypatch):
    use_service(monkeypatch)
    response = client.post(
        "/api/ask/stream",
        json={"question": "A mousse", "mode": "advisor", "top_k": 3},
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    events = [json.loads(line) for line in response.text.splitlines()]
    assert [event["type"] for event in events] == [
        "status",
        "evidence",
        "token",
        "token",
        "done",
    ]
    assert events[1]["products"][0]["name"] == "Mousse"
    assert events[-1]["answer"] == "Use the mousse."


def test_ask_validation(monkeypatch):
    use_service(monkeypatch)
    for payload in (
        {"question": "   "},
        {"question": "hello", "mode": "invalid"},
        {"question": "hello", "top_k": 0},
        {"question": "hello", "top_k": 26},
    ):
        assert client.post("/api/ask", json=payload).status_code == 422


def test_service_error_is_safe_503(monkeypatch):
    fake = FakeService()

    def fail(*_args, **_kwargs):
        raise ServiceError("ollama_unavailable", "Ollama is unavailable.", "Start Ollama.")

    fake.ask = fail
    use_service(monkeypatch, fake)
    response = client.post("/api/ask", json={"question": "hello"})
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "ollama_unavailable"
    assert "traceback" not in response.text.lower()


def test_missing_database_and_vector_index_are_503(monkeypatch):
    for code in ("database_missing", "vector_index_missing"):
        fake = FakeService()

        def fail(*_args, _code=code, **_kwargs):
            raise ServiceError(_code, "Setup is incomplete.", "Run the setup command.")

        fake.ask = fail
        use_service(monkeypatch, fake)
        response = client.post("/api/ask", json={"question": "hello"})
        assert response.status_code == 503
        assert response.json()["detail"]["code"] == code


def test_stream_encodes_late_service_error(monkeypatch):
    fake = FakeService()

    def fail(*_args, **_kwargs):
        yield {"type": "status", "stage": "embedding", "message": "Embedding..."}
        raise ServiceError("ollama_unavailable", "Ollama is unavailable.", "Start Ollama.")

    fake.stream_ask = fail
    use_service(monkeypatch, fake)
    response = client.post("/api/ask/stream", json={"question": "hello"})
    events = [json.loads(line) for line in response.text.splitlines()]
    assert response.status_code == 200
    assert events[-1]["type"] == "error"
    assert events[-1]["code"] == "ollama_unavailable"


def test_product_and_missing_product(monkeypatch):
    use_service(monkeypatch)
    found = client.get("/api/products/1")
    assert found.status_code == 200
    assert found.json()["ingredients"] == ["Aqua"]
    assert client.get("/api/products/999").status_code == 404


def test_frontend_and_static_assets_are_served():
    home = client.get("/")
    assert home.status_code == 200
    assert "Kruidvat Ingredient Advisor" in home.text
    javascript = client.get("/static/app.js")
    stylesheet = client.get("/static/styles.css")
    assert javascript.status_code == 200
    assert "innerHTML" not in javascript.text
    assert "renderAnswerMarkdown" in javascript.text
    assert 'createElement("strong")' in javascript.text
    assert 'fetch("/api/ask/stream"' in javascript.text
    assert stylesheet.status_code == 200
