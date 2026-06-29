#!/usr/bin/env python3
"""embed.py: add a semantic embedding for every product in the database.

For each product that has ingredients, we build a short text (name +
ingredients), turn it into a vector with a local Ollama embedding model,
and store that vector in the SAME SQLite file using the sqlite-vec
extension. Once this has run, the catalogue can be searched semantically.

Run it once after scraping. Re-running only embeds NEW products, so it's
safe to run again after each scrape.

Setup (one time):
    pip install -r requirements.txt   # includes sqlite-vec
    ollama pull nomic-embed-text      # the embedding model
    # make sure `ollama serve` is running

Usage:
    python embed.py --db kruidvat.db
"""

import argparse
import json
import sqlite3
from urllib import request as urllib_request

import sqlite_vec
from sqlite_vec import serialize_float32

# --- defaults (override via CLI) ---------------------------------------------
DB_DEFAULT = "kruidvat.db"          # the database scraper.py writes to
OLLAMA_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"    # 768-dim embeddings; good local default
EMBED_DIM = 768                     # must match the model's output size
OLLAMA_TIMEOUT = 60.0
# -----------------------------------------------------------------------------


def get_embedding(text, *, model, url, timeout):
    """Ask the local Ollama model to turn `text` into an embedding vector.

    Uses urllib (no external HTTP dependency) to match the rest of the project.
    """
    payload = {"model": model, "prompt": text}
    req = urllib_request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib_request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", errors="ignore")
    return json.loads(body)["embedding"]


def ingredients_to_text(ingredients_list):
    """`ingredients_list` is stored as a JSON array string by the scraper.

    Turn it back into a readable comma-separated string for embedding, and
    fall back to the raw value if it isn't valid JSON.
    """
    try:
        items = json.loads(ingredients_list)
        if isinstance(items, list):
            return ", ".join(str(i) for i in items)
    except Exception:
        pass
    return ingredients_list


def build_text(name, ingredients):
    """The text we embed: the name carries the product type, the ingredients
    carry the chemical detail, so search matches on both."""
    return f"{name}\nIngredients: {ingredients}"


def main():
    parser = argparse.ArgumentParser(
        description="Embed scraped products into a sqlite-vec table for semantic search."
    )
    parser.add_argument(
        "--db", default=DB_DEFAULT, help="SQLite database file written by scraper.py"
    )
    parser.add_argument(
        "--embed-model", default=EMBED_MODEL, help="Local Ollama embedding model"
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=EMBED_DIM,
        help="Embedding size (must match the model's output)",
    )
    parser.add_argument(
        "--ollama-url", default=OLLAMA_URL, help="Ollama embeddings endpoint"
    )
    parser.add_argument("--ollama-timeout", type=float, default=OLLAMA_TIMEOUT)
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)

    # load the sqlite-vec extension into this connection
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    # a vector table keyed by the product id from the `products` table
    conn.execute(
        f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_products
        USING vec0(product_id INTEGER PRIMARY KEY, embedding FLOAT[{args.embed_dim}])
        """
    )

    # which products are already embedded? (so re-runs skip them)
    already_done = {
        row[0] for row in conn.execute("SELECT product_id FROM vec_products")
    }

    # only embed products that actually have ingredients
    rows = conn.execute(
        """
        SELECT id, name, ingredients_list
        FROM products
        WHERE ingredients_list IS NOT NULL AND TRIM(ingredients_list) != ''
        """
    ).fetchall()

    todo = [r for r in rows if r[0] not in already_done]
    print(f"{len(rows)} products with ingredients | {len(todo)} new to embed.")

    embedded = 0
    for i, (pid, name, ingredients_list) in enumerate(todo, start=1):
        text = build_text(name or "", ingredients_to_text(ingredients_list))
        try:
            vector = get_embedding(
                text,
                model=args.embed_model,
                url=args.ollama_url,
                timeout=args.ollama_timeout,
            )
            conn.execute(
                "INSERT INTO vec_products(product_id, embedding) VALUES (?, ?)",
                (pid, serialize_float32(vector)),
            )
            conn.commit()
            embedded += 1
            print(f"[{i}/{len(todo)}] embedded: {name}")
        except Exception as e:
            print(f"[{i}/{len(todo)}] FAILED  {name}: {e}")

    conn.close()
    print(f"Done. Embedded {embedded} new product(s).")


if __name__ == "__main__":
    main()
