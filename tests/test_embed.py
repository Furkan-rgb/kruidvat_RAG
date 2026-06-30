"""Unit tests for embed.py helpers (no Ollama: HTTP is mocked)."""

import json

import embed


def test_ingredients_to_text_json_list():
    assert embed.ingredients_to_text('["Aqua", "Glycerin"]') == "Aqua, Glycerin"


def test_ingredients_to_text_invalid_json_returns_raw():
    assert embed.ingredients_to_text("Aqua, Glycerin") == "Aqua, Glycerin"


def test_ingredients_to_text_non_list_json_returns_raw():
    assert embed.ingredients_to_text('{"a": 1}') == '{"a": 1}'


def test_build_text_contains_name_description_and_ingredients():
    t = embed.build_text("Shampoo", "Gentle daily shampoo.", "Aqua, Glycerin")
    assert "Shampoo" in t
    assert "Gentle daily shampoo." in t
    assert "Aqua" in t and "Ingredients:" in t


def test_build_text_skips_empty_description():
    t = embed.build_text("Shampoo", "", "Aqua")
    assert t == "Shampoo\nIngredients: Aqua"


def test_get_embedding_builds_request_and_parses(monkeypatch):
    captured = {}

    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"embedding": [0.1, 0.2, 0.3]}).encode()

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode())
        captured["timeout"] = timeout
        return FakeResp()

    monkeypatch.setattr(embed.urllib_request, "urlopen", fake_urlopen)

    vec = embed.get_embedding(
        "hello", model="embeddinggemma", url="http://x/api/embeddings", timeout=12
    )
    assert vec == [0.1, 0.2, 0.3]
    assert captured["url"] == "http://x/api/embeddings"
    assert captured["body"] == {"model": "embeddinggemma", "prompt": "hello"}
    assert captured["timeout"] == 12
