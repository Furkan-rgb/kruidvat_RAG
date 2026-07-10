#!/usr/bin/env python3
"""Semantic search and grounded Q&A over the scraped Kruidvat catalogue.

The reusable pipeline lives in :mod:`app.service`; this module remains the
backwards-compatible command-line entry point and re-exports its established
helper functions and prompt constants.
"""

import argparse

import config
from app import service as _service
from app.service import (
    ADVISOR_SYSTEM_PROMPT,
    CLOSING_INSTRUCTIONS,
    GROUNDED_SYSTEM_PROMPT,
    SYSTEM_PROMPTS,
    RAGService,
    _post_json,
    build_context,
    ingredients_to_text,
    search,
)

__all__ = [
    "ADVISOR_SYSTEM_PROMPT",
    "CLOSING_INSTRUCTIONS",
    "GROUNDED_SYSTEM_PROMPT",
    "SYSTEM_PROMPTS",
    "build_context",
    "generate_answer",
    "get_embedding",
    "ingredients_to_text",
    "search",
]


def get_embedding(text, *, model, url, timeout):
    """Compatibility wrapper around the shared embedding helper."""
    return _service.get_embedding(
        text, model=model, url=url, timeout=timeout, _post=_post_json
    )


def generate_answer(question, context, *, provider, model, url, timeout, mode="advisor"):
    """Compatibility wrapper around the shared answer helper."""
    return _service.generate_answer(
        question,
        context,
        provider=provider,
        model=model,
        url=url,
        timeout=timeout,
        mode=mode,
        _post=_post_json,
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
        "--answer-model", default=config.ANSWER_MODEL, help="Model used to write the answer"
    )
    parser.add_argument("--embed-url", default=config.EMBEDDINGS_URL)
    parser.add_argument("--generate-url", default=config.GENERATE_URL)
    parser.add_argument("--ollama-timeout", type=float, default=config.OLLAMA_TIMEOUT)
    args = parser.parse_args()

    service = RAGService(
        db_path=args.db,
        embed_model=args.embed_model,
        answer_model=args.answer_model,
        provider=args.provider,
        embed_url=args.embed_url,
        generate_url=args.generate_url,
        timeout=args.ollama_timeout,
    )
    question, rows = service.retrieve(args.question, mode=args.mode, top_k=args.top_k)

    if not rows:
        print("No matches found. Have you run embed.py on this database yet?")
        return

    print(f"\nTop {len(rows)} matches:")
    for dist, _pid, name, url, _desc, _ing in rows:
        print(f"  [{dist:.3f}] {name}  {url}")

    if not args.search:
        answer = generate_answer(
            question,
            build_context(rows),
            provider=args.provider,
            model=args.answer_model,
            url=args.generate_url,
            timeout=args.ollama_timeout,
            mode=args.mode,
        )
        print("\nAnswer:\n" + answer)


if __name__ == "__main__":
    main()
