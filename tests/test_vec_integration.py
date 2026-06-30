"""End-to-end retrieval test: embed products, then search, through real sqlite-vec.

Ollama is replaced with deterministic fake embeddings, so this proves the
sqlite-vec wiring and the KNN query in query.search() without any model server.
Skipped automatically if sqlite-vec can't load in this Python build.
"""

import json
import sqlite3

import pytest

import db
import embed
import query


def _sqlite_vec_loadable():
    try:
        import sqlite_vec

        c = sqlite3.connect(":memory:")
        c.enable_load_extension(True)
        sqlite_vec.load(c)
        c.execute("select vec_version()")
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _sqlite_vec_loadable(),
    reason="sqlite-vec extension not loadable in this Python build",
)


def _fake_vec(text):
    """4-dim one-hot by product type, so nearest-neighbour is unambiguous."""
    t = text.lower()
    if "shampoo" in t:
        return [1.0, 0.0, 0.0, 0.0]
    if "conditioner" in t:
        return [0.0, 1.0, 0.0, 0.0]
    if "hairspray" in t:
        return [0.0, 0.0, 1.0, 0.0]
    return [0.0, 0.0, 0.0, 1.0]


def _open_vec(path):
    conn = sqlite3.connect(path)
    conn.enable_load_extension(True)
    import sqlite_vec

    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def test_embed_then_search(tmp_path, monkeypatch):
    dbpath = str(tmp_path / "kv.db")

    conn = db.setup_db(dbpath)
    db.save_products_batch(
        conn,
        [
            ("Nice Shampoo", "https://x/p/1", json.dumps(["Aqua", "SLS"]), "t"),
            ("Soft Conditioner", "https://x/p/2", json.dumps(["Aqua", "Cetearyl"]), "t"),
            ("Strong Hairspray", "https://x/p/3", json.dumps(["Alcohol Denat"]), "t"),
        ],
    )
    conn.close()

    # embed with fake vectors (no Ollama), small dimension
    monkeypatch.setattr(embed, "get_embedding", lambda text, **kw: _fake_vec(text))
    monkeypatch.setattr("sys.argv", ["embed.py", "--db", dbpath, "--embed-dim", "4"])
    embed.main()

    conn = _open_vec(dbpath)
    assert conn.execute("SELECT COUNT(*) FROM vec_products").fetchone()[0] == 3

    # query for shampoo -> closest product must be the shampoo
    rows = query.search(conn, _fake_vec("shampoo"), top_k=3)
    assert rows, "KNN returned no rows"
    assert rows[0][2] == "Nice Shampoo"  # (distance, id, name, url, ingredients)
    # distances are sorted ascending
    distances = [r[0] for r in rows]
    assert distances == sorted(distances)
    conn.close()


def test_reset_rebuilds_vector_table(tmp_path, monkeypatch):
    dbpath = str(tmp_path / "kv.db")
    conn = db.setup_db(dbpath)
    db.save_products_batch(
        conn, [("Nice Shampoo", "https://x/p/1", json.dumps(["Aqua"]), "t")]
    )
    conn.close()

    monkeypatch.setattr(embed, "get_embedding", lambda text, **kw: _fake_vec(text))
    monkeypatch.setattr("sys.argv", ["embed.py", "--db", dbpath, "--embed-dim", "4"])
    embed.main()

    # second run with --reset should not error and should still leave 1 row
    monkeypatch.setattr(
        "sys.argv", ["embed.py", "--db", dbpath, "--embed-dim", "4", "--reset"]
    )
    embed.main()

    conn = _open_vec(dbpath)
    assert conn.execute("SELECT COUNT(*) FROM vec_products").fetchone()[0] == 1
    conn.close()
