#!/usr/bin/env python3
"""query.py: ask questions about the scraped catalogue (semantic RAG).

This is the payoff of the pipeline. It embeds your question with the same
local model used in embed.py, finds the closest products in the sqlite-vec
table, and hands those products to a local LLM.

The answer step has two modes (--mode). "advisor" (the default) combines the
retrieved product data with general haircare knowledge to make a
recommendation, and stays explicit about which is catalogue fact and which is
its own judgement. "strict" stays inside the retrieved context and only reports
what the ingredient lists literally say. Either way the product facts come from
real, current catalogue data, not the model's memory.

Embedding and retrieval are always local. The final answer step is pluggable
via ANSWER_PROVIDER in config.py (local "ollama" for now); a remote model can
be added later as one extra branch in generate_answer().

Examples:
    # Advisor: recommend, using the products plus haircare knowledge
    python query.py "Which mousse is best for fine, flat hair?"

    # Strict: answer only from the retrieved ingredient lists
    python query.py --mode strict "Which of these are alcohol-free?"

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

GROUNDED_SYSTEM_PROMPT = """You answer questions about cosmetic products using ONLY the product data provided in the context.

Each product comes with its full INCI ingredient list. Reason over those lists:
- "Is it <X>-free?" (e.g. sulfate-free, paraben-free, silicone-free): a product IS <X>-free when no ingredient in its list is <X> or a known variant of it. Examples of variants:
  - sulfates: Sodium Laureth Sulfate, Sodium Lauryl Sulfate, Ammonium Lauryl/Laureth Sulfate, Sodium Coco-Sulfate, TEA-Lauryl Sulfate.
  - parabens: any ingredient ending in "paraben".
  - silicones: Dimethicone, and ingredients ending in "-cone" / "-siloxane".
- "Does it contain <X>?": true only if <X> (or a variant) appears in its ingredient list.

Rules:
1. Use only the products and ingredient lists in the context; do not rely on outside knowledge about specific products.
2. Only say you cannot answer if the needed ingredient lists are missing or empty -- not when you simply have to read them.
3. Name products using their exact names from the context, and briefly say why (e.g. "no sulfate listed").
4. Be concise and factual.
"""

ADVISOR_SYSTEM_PROMPT = """You are a knowledgeable haircare and cosmetics advisor. You help someone choose from a shortlist of catalogue products given in the context. Each product comes with its name, an optional description, and its full INCI ingredient list.

Give a genuinely useful recommendation, not a literal lookup. Combine two sources:
1. The product data in the context (descriptions and ingredient lists). These are the only real products and ingredients you may discuss.
2. General haircare and cosmetics knowledge: what ingredients do, and which hair or skin types suit which kinds of formula. Use this to interpret the data.

Grounding rules (critical, these prevent mistakes):
- Each ingredient list belongs to one product only. If you name an ingredient as a reason, it MUST appear in THAT product's own "Ingredients of <product>" line in the context. Never carry an ingredient over from another product, and never add one from general knowledge of the brand or product.
- Before you state that a product contains an ingredient, find that exact ingredient in its own list. If it is not there, do not mention it; reason from that product's description instead.
- Do not invent a product, a description, or an ingredient that is not in the context.

How to reason:
- Read each product's own ingredients and description, then apply general principles to judge fit. For example: fine hair needs lightweight products, so styling products that actually list heavy butters or oils (such as Butyrospermum Parkii / Shea Butter, Ricinus Communis / castor oil, Cocos Nucifera / coconut oil) tend to weigh it down, while light water-based formulas suit it better. Only invoke such an ingredient for a product if it is in that product's list.
- Recommend specific products by their exact names from the context, rank them when that helps, and justify each with its own listed ingredients or its description.
- If the question asks about something the catalogue does not record (for example hair porosity, or a curl type like "2c"), say so plainly, then still give your best recommendation from what the ingredients and descriptions do tell you. Do not refuse just because the exact attribute is missing.

Be honest about what is fact and what is judgement:
- What is in a product is fact, taken from that product's own list in the context.
- The general principles are your own knowledge.
- The recommendation is you applying those principles to the product. Do not state a principle-based inference as if the catalogue said it.

Be concise and practical: a short ranked recommendation with one-line reasons beats a long essay.
"""

# Map --mode to its system prompt and the closing instruction in the user turn.
SYSTEM_PROMPTS = {"advisor": ADVISOR_SYSTEM_PROMPT, "strict": GROUNDED_SYSTEM_PROMPT}
CLOSING_INSTRUCTIONS = {
    "advisor": (
        "Recommend from the products above. Combine their ingredients and "
        "descriptions with general haircare knowledge, and keep clear what is "
        "catalogue fact and what is your own judgement."
    ),
    "strict": "Answer using only the context above.",
}


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
    (distance, id, name, url, description, ingredients_list), closest first."""
    return conn.execute(
        """
        WITH matches AS (
            SELECT product_id, distance
            FROM vec_products
            WHERE embedding MATCH ? AND k = ?
        )
        SELECT m.distance, p.id, p.name, p.url, p.description, p.ingredients_list
        FROM matches AS m
        JOIN products AS p ON p.id = m.product_id
        ORDER BY m.distance
        """,
        (serialize_float32(query_vector), top_k),
    ).fetchall()


def build_context(rows):
    """Turn retrieved products into a compact, grounded context block.

    Each product is numbered and its ingredient line is tagged with the product
    name ("Ingredients of <name>:"), so the model cannot blur one product's
    ingredients into another's when it reasons across the shortlist.
    """
    blocks = []
    for i, (_dist, _pid, name, url, description, ingredients_list) in enumerate(rows, 1):
        block = f"Product {i}: {name}\nURL: {url}\n"
        if description:
            block += f"Description: {description}\n"
        block += f"Ingredients of {name}: {ingredients_to_text(ingredients_list)}"
        blocks.append(block)
    return "\n\n----------\n\n".join(blocks)


def _answer_ollama(prompt, *, model, url, timeout, system):
    """Local Ollama backend for the answer step."""
    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "system": system,
        "options": {"temperature": 0},
        "prompt": prompt,
    }
    return _post_json(url, payload, timeout).get("response", "").strip()


def generate_answer(question, context, *, provider, model, url, timeout, mode="advisor"):
    """Write the final answer from the retrieved context.

    `mode` selects how the answer is written: "advisor" combines the retrieved
    products with general haircare knowledge to recommend; "strict" stays inside
    the context (see SYSTEM_PROMPTS / CLOSING_INSTRUCTIONS). Embedding and
    retrieval are always local; only this step's backend is pluggable. To add a
    remote model later (e.g. "anthropic"), add a branch below that reads its API
    key from the environment and calls the provider. Nothing else changes.
    """
    system = SYSTEM_PROMPTS[mode]
    prompt = (
        f"Context (retrieved products):\n{context}\n\n"
        f"Question: {question}\n\n"
        f"{CLOSING_INSTRUCTIONS[mode]}"
    )
    if provider == "ollama":
        return _answer_ollama(prompt, model=model, url=url, timeout=timeout, system=system)
    # elif provider == "anthropic":
    #     return _answer_anthropic(prompt, model=model, system=system, timeout=timeout)
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
        "--mode",
        choices=["advisor", "strict"],
        default=config.ANSWER_MODE,
        help="advisor: recommend, combining products with haircare knowledge; "
        "strict: answer only from the retrieved context",
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
    for dist, _pid, name, url, _desc, _ing in rows:
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
            mode=args.mode,
        )
        print("\nAnswer:\n" + answer)

    conn.close()


if __name__ == "__main__":
    main()
