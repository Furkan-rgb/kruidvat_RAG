"""End-to-end retrieval test: embed products, then search, through real sqlite-vec.

Ollama is replaced with deterministic fake embeddings, so this proves the
sqlite-vec wiring and the KNN query in query.search() without any model server.
Skipped automatically if sqlite-vec can't load in this Python build.
"""

import json
import sqlite3

import pytest

from lib import db
from app import service
from app.index_metadata import read_index_metadata, write_index_metadata
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


class FakeEmbeddingProvider:
    provider = "fake"
    model = "fake-model"
    document_prefix = ""
    query_prefix = ""

    def __init__(self, dimension):
        self.dimension = dimension

    def embed_documents(self, texts):
        return [_fake_vec(text) for text in texts]

    def embed_query(self, text):
        return _fake_vec(text)


def use_fake_embedding_provider(monkeypatch):
    monkeypatch.setattr(
        embed,
        "create_embedding_provider",
        lambda _provider, **kwargs: FakeEmbeddingProvider(kwargs["dimension"]),
    )


def test_embed_then_search(tmp_path, monkeypatch):
    dbpath = str(tmp_path / "kv.db")

    conn = db.setup_db(dbpath)
    db.save_products_batch(
        conn,
        [
            ("Nice Shampoo", "https://x/p/1", "", json.dumps(["Aqua", "SLS"]), "t"),
            ("Soft Conditioner", "https://x/p/2", "", json.dumps(["Aqua", "Cetearyl"]), "t"),
            ("Strong Hairspray", "https://x/p/3", "", json.dumps(["Alcohol Denat"]), "t"),
        ],
    )
    conn.close()

    # embed with fake vectors (no Ollama), small dimension
    use_fake_embedding_provider(monkeypatch)
    monkeypatch.setattr("sys.argv", ["embed.py", "--db", dbpath, "--embed-dim", "4"])
    embed.main()

    conn = _open_vec(dbpath)
    assert conn.execute("SELECT COUNT(*) FROM vec_products").fetchone()[0] == 3
    metadata = read_index_metadata(conn)
    assert metadata is not None
    assert metadata["provider"] == "fake"
    assert metadata["model"] == "fake-model"
    assert metadata["dimension"] == 4

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
        conn, [("Nice Shampoo", "https://x/p/1", "", json.dumps(["Aqua"]), "t")]
    )
    conn.close()

    use_fake_embedding_provider(monkeypatch)
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


def test_service_rejects_embedding_profile_mismatch(tmp_path):
    dbpath = str(tmp_path / "kv.db")
    conn = db.setup_db(dbpath)
    conn.enable_load_extension(True)
    import sqlite_vec

    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute(
        "CREATE VIRTUAL TABLE vec_products "
        "USING vec0(product_id INTEGER PRIMARY KEY, embedding FLOAT[4])"
    )
    write_index_metadata(conn, FakeEmbeddingProvider(4))
    conn.commit()
    conn.close()

    configured = FakeEmbeddingProvider(4)
    configured.model = "different-model"
    rag = service.RAGService(
        db_path=dbpath,
        embedding_provider=configured,
        answer_provider=service.OllamaAnswerProvider(model="answer", url="u", timeout=1),
    )
    with pytest.raises(service.ServiceError) as caught:
        rag._connect(require_vector=True)
    assert caught.value.code == "embedding_index_mismatch"


def test_legacy_index_without_metadata_remains_usable(tmp_path):
    dbpath = str(tmp_path / "kv.db")
    conn = db.setup_db(dbpath)
    conn.enable_load_extension(True)
    import sqlite_vec

    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute(
        "CREATE VIRTUAL TABLE vec_products "
        "USING vec0(product_id INTEGER PRIMARY KEY, embedding FLOAT[4])"
    )
    conn.commit()
    conn.close()

    rag = service.RAGService(
        db_path=dbpath,
        embedding_provider=FakeEmbeddingProvider(4),
        answer_provider=service.OllamaAnswerProvider(model="answer", url="u", timeout=1),
    )
    connected = rag._connect(require_vector=True)
    connected.close()


def test_embed_requires_explicit_adoption_before_mutating_legacy_index(
    tmp_path, monkeypatch
):
    dbpath = str(tmp_path / "kv.db")
    conn = db.setup_db(dbpath)
    db.save_products_batch(
        conn, [("Shampoo", "https://x/p/1", "", json.dumps(["Aqua"]), "t")]
    )
    conn.close()
    use_fake_embedding_provider(monkeypatch)
    monkeypatch.setattr("sys.argv", ["embed.py", "--db", dbpath, "--embed-dim", "4"])
    embed.main()

    conn = _open_vec(dbpath)
    conn.execute("DELETE FROM embedding_index_metadata")
    conn.commit()
    conn.close()

    with pytest.raises(SystemExit):
        embed.main()

    monkeypatch.setattr(
        "sys.argv",
        [
            "embed.py",
            "--db",
            dbpath,
            "--embed-dim",
            "4",
            "--adopt-legacy-index",
        ],
    )
    embed.main()
    conn = _open_vec(dbpath)
    assert read_index_metadata(conn)["model"] == "fake-model"
    conn.close()
