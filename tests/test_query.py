"""Unit tests for query.py helpers and the answer-provider dispatch (HTTP mocked)."""

import pytest

import query


def test_ingredients_to_text_handles_none():
    assert query.ingredients_to_text(None) == ""


def test_build_context_formats_each_product():
    rows = [(0.12, 1, "Shampoo", "https://x/p/1", "Gentle daily shampoo.", '["Aqua"]')]
    ctx = query.build_context(rows)
    assert "Product: Shampoo" in ctx
    assert "URL: https://x/p/1" in ctx
    assert "Description: Gentle daily shampoo." in ctx
    assert "Ingredients: Aqua" in ctx


def test_generate_answer_unknown_provider_raises():
    with pytest.raises(ValueError):
        query.generate_answer(
            "q", "ctx", provider="banana", model="m", url="u", timeout=1
        )


def test_generate_answer_ollama_path(monkeypatch):
    captured = {}

    def fake_post(url, payload, timeout):
        captured["url"] = url
        captured["payload"] = payload
        return {"response": "  Final answer.  "}

    monkeypatch.setattr(query, "_post_json", fake_post)

    out = query.generate_answer(
        "Does it contain Aqua?",
        "Product: X\nIngredients: Aqua",
        provider="ollama",
        model="gemma4:e4b",
        url="http://x/api/generate",
        timeout=30,
    )
    assert out == "Final answer."  # stripped
    assert captured["payload"]["model"] == "gemma4:e4b"
    # both the retrieved context and the question must reach the model
    assert "Aqua" in captured["payload"]["prompt"]
    assert "Does it contain Aqua?" in captured["payload"]["prompt"]


def test_mode_selects_system_prompt_and_closing(monkeypatch):
    captured = {}

    def fake_post(url, payload, timeout):
        captured["system"] = payload["system"]
        captured["prompt"] = payload["prompt"]
        return {"response": "ok"}

    monkeypatch.setattr(query, "_post_json", fake_post)

    query.generate_answer(
        "q", "ctx", provider="ollama", model="m", url="u", timeout=1, mode="advisor"
    )
    assert captured["system"] == query.ADVISOR_SYSTEM_PROMPT
    assert "recommend" in captured["prompt"].lower()

    query.generate_answer(
        "q", "ctx", provider="ollama", model="m", url="u", timeout=1, mode="strict"
    )
    assert captured["system"] == query.GROUNDED_SYSTEM_PROMPT
    assert "only the context" in captured["prompt"]


def test_generate_answer_defaults_to_advisor(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        query, "_post_json", lambda url, payload, timeout: captured.update(payload) or {"response": "ok"}
    )
    query.generate_answer("q", "ctx", provider="ollama", model="m", url="u", timeout=1)
    assert captured["system"] == query.ADVISOR_SYSTEM_PROMPT


def test_get_embedding_delegates_to_post_json(monkeypatch):
    monkeypatch.setattr(
        query, "_post_json", lambda url, payload, timeout: {"embedding": [1.0, 2.0]}
    )
    assert query.get_embedding("hi", model="m", url="u", timeout=1) == [1.0, 2.0]
