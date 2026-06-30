"""Unit tests for db.py persistence (real SQLite, temp file, no external deps)."""

import db


def test_setup_and_roundtrip(tmp_path):
    conn = db.setup_db(str(tmp_path / "t.db"))
    db.save_products_batch(
        conn,
        [
            ("Shampoo", "https://x/p/1", '["Aqua"]', "2026-01-01T00:00:00Z"),
            ("Conditioner", "https://x/p/2", '["Glycerin"]', "2026-01-01T00:00:00Z"),
        ],
    )
    assert db.read_existing_urls(conn) == {"https://x/p/1", "https://x/p/2"}


def test_url_unique_insert_or_ignore(tmp_path):
    conn = db.setup_db(str(tmp_path / "t.db"))
    db.save_products_batch(conn, [("A", "https://x/p/1", "[]", "t")])
    db.save_products_batch(conn, [("A2", "https://x/p/1", "[]", "t")])  # dup url
    assert conn.execute("SELECT COUNT(*) FROM products").fetchone()[0] == 1
    # first write wins under INSERT OR IGNORE
    name = conn.execute(
        "SELECT name FROM products WHERE url=?", ("https://x/p/1",)
    ).fetchone()[0]
    assert name == "A"


def test_empty_batch_is_noop(tmp_path):
    conn = db.setup_db(str(tmp_path / "t.db"))
    db.save_products_batch(conn, [])
    assert db.read_existing_urls(conn) == set()
