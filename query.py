#!/usr/bin/env python3
"""query.py: ask questions about the scraped catalogue (semantic RAG).

This is the payoff of the pipeline. It embeds your question with the same
local model used in embed.py, finds the closest products in the sqlite-vec
table, and (by default) hands those products to a local LLM as grounding, so
the answer comes from real, current ingredient data instead of the model's
memory.

Embedding and retrieval are always local. The final answer step is pluggable
via ANSWER_PROVIDER in config.py (local "ollama" for now); a remote model can
be added later as one extra branch in generate_answer().

Examples:
    # Ground a local LLM and get an answer
    python query.py "Which hairsprays are alcohol-free?"

    # Just retrieve the most relevant products, no LLM
    python query.py --search "contains Linalool"

Setup:
    pip install -r requirements.txt   # includes sqlite-vec
    ollama pull embeddinggemma        # embeddings (same as embed.py)
    ollama pull gemma4:e4b            # answer generation
    # make sure `ollama serve` is running
"""

import argparse
import json
import sqlite3
from urllib import request as urllib_request

import sqlite_vec
from sqlite_vec import serialize_float32

import config

ANSWER_SYSTEM_PROMPT = """You answer questions about cosmetic products using ONLY the product data provided in the context.

Rules:
1. Use only the products and ingredients listed in the context. Do not rely on outside knowledge or memory.
2. If the context does not contain the answer, say so plainly instead of guessing.
3. When you name a product, use its exact name from the context.
4. Be concise and factual.
"""


def _post_json(url, payload, timeout):
    """POST a JSON payload to a local Ollama endpoint and return the parsed body."""
    req = urllib_request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib_request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", errors="ignore")
    return json.loads(body)


def get_embedding(text, *, model, url, timeout):
    """Embed `text` with the local Ollama embedding model."""
    return _post_json(url, {"model": model, "prompt": text}, timeout)["embedding"]


def ingredients_to_text(ingredients_list):
    """`ingredients_list` is stored as a JSON array string by the scraper."""
    try:
        items = json.loads(ingredients_list)
        if isinstance(items, list):
            return ", ".join(str(i) for i in items)
    except Exception:
        pass
    return ingredients_list or ""


def search(conn, query_vector, top_k):
    """Return the `top_k` nearest products as
    (distance, id, name, url, ingredients_list), closest first."""
    return conn.execute(
        """
        WITH matches AS (
            SELECT product_id, distance
            FROM vec_products
            WHERE embedding MATCH ? AND k = ?
        )
        SELECT m.distance, p.id, p.name, p.url, p.ingredients_list
        FROM matches AS m
        JOIN products AS p ON p.id = m.product_id
        ORDER BY m.distance
        """,
        (serialize_float32(query_vector), top_k),
    ).fetchall()


def build_context(rows):
    """Turn retrieved products into a compact, grounded context block."""
    blocks = []
    for _dist, _pid, name, url, ingredients_list in rows:
        blocks.append(
            f"Product: {name}\n"
            f"URL: {url}\n"
            f"Ingredients: {ingredients_to_text(ingredients_list)}"
        )
    return "\n\n".join(blocks)


def _answer_ollama(prompt, *, model, url, timeout):
    """Local Ollama backend for the answer step."""
    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "system": ANSWER_SYSTEM_PROMPT,
        "options": {"temperature": 0},
        "prompt": prompt,
    }
    return _post_json(url, payload, timeout).get("response", "").strip()


def generate_answer(question, context, *, provider, model, url, timeout):
    """Write the final answer from the retrieved context.

    Embedding and retrieval are always local; only this step's backend is
    pluggable. To add a remote model later (e.g. "anthropic"), add a branch
    below that reads its API key from the environment and calls the provider.
    Nothing else in the pipeline needs to change.
    """
    prompt = (
        f"Context (retrieved products):\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer using only the context above."
    )
    if provider == "ollama":
        return _answer_ollama(prompt, model=model, url=url, timeout=timeout)
    # elif provider == "anthropic":
    #     return _answer_anthropic(prompt, model=model, timeout=timeout)
    raise ValueError(
        f"Unknown answer provider {provider!r}. Implemented: 'ollama'. "
        "Add a branch in generate_answer() to support a remote model."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Semantic search and grounded Q&A over the scraped catalogue."
    )
    parser.add_argument(
        "question", help="Your question (or search text when using --search)"
    )
    parser.add_argument("--db", default=config.DB_PATH, help="SQLite database to query")
    parser.add_argument(
        "--search",
        action="store_true",
        help="Only retrieve and print the nearest products (skip the LLM answer)",
    )
    parser.add_argument(
        "--top-k", type=int, default=config.TOP_K, help="Number of products to retrieve"
    )
    parser.add_argument(
        "--provider",
        default=config.ANSWER_PROVIDER,
        help="Backend that writes the answer (currently: ollama)",
    )
    parser.add_argument(
        "--embed-model",
        default=config.EMBED_MODEL,
        help="Embedding model (must match embed.py)",
    )
    parser.add_argument(
        "--answer-model",
        default=config.ANSWER_MODEL,
        help="Model used to write the answer",
    )
    parser.add_argument("--embed-url", default=config.EMBEDDINGS_URL)
    parser.add_argument("--generate-url", default=config.GENERATE_URL)
    parser.add_argument("--ollama-timeout", type=float, default=config.OLLAMA_TIMEOUT)
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    # Prefix with the model's query instruction (see config.EMBED_QUERY_PREFIX)
    # so the question is embedded the way the documents were in embed.py.
    query_vector = get_embedding(
        config.EMBED_QUERY_PREFIX + args.question,
        model=args.embed_model,
        url=args.embed_url,
        timeout=args.ollama_timeout,
    )
    rows = search(conn, query_vector, args.top_k)

    if not rows:
        print("No matches found. Have you run embed.py on this database yet?")
        conn.close()
        return

    print(f"\nTop {len(rows)} matches:")
    for dist, _pid, name, url, _ing in rows:
        print(f"  [{dist:.3f}] {name}  {url}")

    if not args.search:
        context = build_context(rows)
        answer = generate_answer(
            args.question,
            context,
            provider=args.provider,
            model=args.answer_model,
            url=args.generate_url,
            timeout=args.ollama_timeout,
        )
        print("\nAnswer:\n" + answer)

    conn.close()


if __name__ == "__main__":
    main()
