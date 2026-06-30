import sqlite3

DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS products (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    url TEXT UNIQUE,
    ingredients_list TEXT,
    scraped_at TEXT
);
"""


def setup_db(path):
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(DB_SCHEMA)
    conn.commit()
    return conn


def save_products_batch(conn, rows):
    if not rows:
        return
    cur = conn.cursor()
    try:
        cur.executemany(
            "INSERT OR IGNORE INTO products (name, url, ingredients_list, scraped_at) VALUES (?, ?, ?, ?)",
            rows,
        )
        conn.commit()
    except Exception as e:
        print("DB batch save error:", e)


def read_existing_urls(conn):
    cur = conn.cursor()
    cur.execute("SELECT url FROM products")
    return {r[0] for r in cur.fetchall()}
