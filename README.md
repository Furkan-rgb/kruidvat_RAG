# Kruidvat scraper

This project scrapes product pages from Kruidvat (category: haarstylingproducten) and saves product name, ingredients and URL into a local SQLite database (`kruidvat.db`).

Setup

1. Create a Python 3.9+ virtual environment and activate it.
2. Install dependencies:

```bash
pip install -r requirements.txt
python -m playwright install
```

Run

```bash
python scraper.py --category "https://www.kruidvat.nl/verzorging/haarstylingproducten" --db kruidvat.db
```

Options

- `--headed` : show browser window (useful for debugging)
- `--max-pages N` : limit pages crawled from category (default 50)
- `--delay S` : delay between requests in seconds (default 1.0)

Notes for RAG / LLM usage

- The scraper saves both `ingredients_raw` (original text) and `ingredients_list` (JSON array of comma-split ingredients). For RAG you'll want both the raw text and a normalized tokenized list.
- Recommended metadata to keep for retrieval: product `name`, `url`, `scraped_at`, and `ingredients_list`.
- To build embeddings later, export rows to a JSONL file (one JSON object per product) and generate embeddings per product record or per ingredient chunk.
