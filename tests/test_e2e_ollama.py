"""Live end-to-end test against a real local Ollama.

Unlike the rest of the suite (which mocks Ollama), this exercises the REAL
pipeline: seed a small catalogue, embed it with a real embedding model, run a
real KNN search, and generate a real grounded answer. It does NOT scrape the
live website -- products are seeded directly so the test stays deterministic.

It SKIPS automatically unless a local Ollama is reachable AND the embedding +
answer models are present. Pull them first (or point at ones you have):

    ollama pull embeddinggemma
    ollama pull gemma4:e4b

Override the models via env vars when you haven't pulled the config defaults:

    E2E_ANSWER_MODEL=qwen3.6:35b-a3b-q4_K_M pytest tests/test_e2e_ollama.py -s
"""

import json
import os
import sqlite3
from urllib import request as urllib_request

import pytest

import config
from lib import db
import embed
import query

EMBED_MODEL = os.environ.get("E2E_EMBED_MODEL", config.EMBED_MODEL)
ANSWER_MODEL = os.environ.get("E2E_ANSWER_MODEL", config.ANSWER_MODEL)


def _ollama_tags():
    try:
        with urllib_request.urlopen(config.OLLAMA_HOST + "/api/tags", timeout=4) as resp:
            data = json.loads(resp.read().decode())
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return None


def _has_model(tags, model):
    base = model.split(":")[0]
    return any(t == model or t.split(":")[0] == base for t in tags)


_TAGS = _ollama_tags()

pytestmark = pytest.mark.skipif(
    _TAGS is None
    or not _has_model(_TAGS, EMBED_MODEL)
    or not _has_model(_TAGS, ANSWER_MODEL),
    reason=(
        "needs a reachable Ollama with the embedding + answer models pulled "
        f"(embed={EMBED_MODEL}, answer={ANSWER_MODEL})"
    ),
)


def _open_vec(path):
    conn = sqlite3.connect(path)
    conn.enable_load_extension(True)
    import sqlite_vec

    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def test_full_pipeline_against_real_ollama(tmp_path, monkeypatch, capsys):
    dbpath = str(tmp_path / "e2e.db")

    # 1) seed a tiny, distinct catalogue (no scraping)
    conn = db.setup_db(dbpath)
    db.save_products_batch(
        conn,
        [
            ("Hydrating Shampoo", "https://x/p/1", "Daily hydrating shampoo.",
             json.dumps(["Aqua", "Sodium Laureth Sulfate", "Glycerin"]), "t"),
            ("Extra Strong Hairspray", "https://x/p/2", "Strong-hold hairspray.",
             json.dumps(["Alcohol Denat", "VP/VA Copolymer", "Parfum"]), "t"),
            ("Nourishing Hair Oil", "https://x/p/3", "Nourishing oil for dry hair.",
             json.dumps(["Argania Spinosa Kernel Oil", "Limonene", "Linalool"]), "t"),
        ],
    )
    conn.close()

    # 2) embed for real with a real Ollama embedding model
    monkeypatch.setattr(
        "sys.argv",
        ["embed.py", "--db", dbpath, "--embed-model", EMBED_MODEL,
         "--embed-dim", str(config.EMBED_DIM), "--reset"],
    )
    embed.main()

    conn = _open_vec(dbpath)
    n = conn.execute("SELECT COUNT(*) FROM vec_products").fetchone()[0]
    assert n == 3, f"expected 3 embedded products, got {n}"

    # 3) real query embedding + KNN search
    question = "Which product contains alcohol?"
    qvec = query.get_embedding(
        config.EMBED_QUERY_PREFIX + question,
        model=EMBED_MODEL,
        url=config.EMBEDDINGS_URL,
        timeout=config.OLLAMA_TIMEOUT,
    )
    assert len(qvec) == config.EMBED_DIM, f"embedding dim {len(qvec)} != {config.EMBED_DIM}"

    rows = query.search(conn, qvec, top_k=3)
    assert rows, "KNN returned no rows"
    distances = [r[0] for r in rows]
    assert distances == sorted(distances), "distances not ascending"

    # 4) real grounded answer (generation can be slow, so allow more time)
    context = query.build_context(rows)
    answer = query.generate_answer(
        question, context, provider="ollama",
        model=ANSWER_MODEL, url=config.GENERATE_URL, timeout=max(config.OLLAMA_TIMEOUT, 180),
    )
    assert isinstance(answer, str) and answer.strip(), "empty answer from model"

    # surface the real output for human inspection (run with `pytest -s`)
    with capsys.disabled():
        print(f"\n--- E2E retrieval (embed={EMBED_MODEL}) ---")
        for d, _id, name, _url, _desc, _ing in rows:
            print(f"  [{d:.3f}] {name}")
        print(f"--- E2E answer ({ANSWER_MODEL}) ---\n{answer}\n")
