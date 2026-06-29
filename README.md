# Kruidvat Ingredient Scraper

A web scraping pipeline that crawls product pages from [Kruidvat](https://www.kruidvat.nl) and extracts clean, structured cosmetic ingredient data (INCI) into a local SQLite database — ready to be embedded for RAG / semantic search.

The interesting part isn't the crawling; it's getting *reliable structured data* out of messy, inconsistent webshop HTML. Instead of brittle CSS-selector parsing, a local LLM reads the sanitized page text and returns a validated ingredient list, while distinguishing real cosmetics from hardware (hair dryers, brushes, etc.) that have no ingredients at all.

## Highlights

- **Async + concurrent** — built on `asyncio` and Playwright; product pages are scraped in parallel with a bounded semaphore.
- **Anti-bot hardening** — `playwright-stealth`, a realistic Chromium fingerprint (locale, timezone, headers, viewport), and automatic cookie/"read more" handling.
- **LLM-based extraction** — a local [Ollama](https://ollama.com) model extracts ingredients from cleaned page text using a carefully engineered, schema-constrained prompt (`{"found": bool, "ingredients": [...]}`), serialized through a single-flight semaphore.
- **Smart pagination** — reads the total product count from the category page and computes exactly how many pages to crawl, with empty-page short-circuiting.
- **Idempotent storage** — SQLite (WAL mode) with batched inserts, URL de-duplication, and skip-already-scraped logic so runs can be resumed.

## Architecture

The code is split into focused, independently testable modules:

| File | Responsibility |
|------|----------------|
| `scraper.py` | CLI entry point and orchestration (crawl → extract → store) |
| `browser.py` | Playwright context/stealth setup, navigation, link & name extraction |
| `pager.py` | Category pagination and product-link collection |
| `extractor.py` | HTML sanitization + LLM ingredient extraction and response parsing |
| `db.py` | SQLite schema and batched persistence |
| `logging_config.py` | Structured JSON logging |

**Flow:** open category page → read total product count → paginate and collect `/p/` product links → filter out already-scraped URLs → concurrently visit each product, sanitize the HTML, and ask the local LLM for ingredients → batch-write results to SQLite.

## Setup

Requires Python 3.9+ and a running [Ollama](https://ollama.com) instance for the extraction step.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
pip install playwright-stealth
python -m playwright install chromium

# 3. Pull the extraction model (default: ministral-3:3b)
ollama pull ministral-3:3b
```

## Usage

```bash
python scraper.py \
  --category "https://www.kruidvat.nl/verzorging/haarstylingproducten" \
  --db kruidvat.db
```

Useful options:

| Flag | Default | Description |
|------|---------|-------------|
| `--headed` | off | Show the browser window (debugging) |
| `--concurrency N` | 10 | Concurrent product pages |
| `--max-pages N` | 200 | Cap on category pages crawled |
| `--limit N` | — | Process at most N products |
| `--delay S` | 0.2 | Delay between paginated requests |
| `--ollama-model` | `ministral-3:3b` | Local Ollama model |
| `--ollama-url` | `localhost:11434` | Ollama generate endpoint |

## Data model

```sql
products(
  id              INTEGER PRIMARY KEY,
  name            TEXT,
  url             TEXT UNIQUE,
  ingredients_list TEXT,   -- JSON array of normalized ingredients
  scraped_at      TEXT     -- ISO 8601 UTC
)
```

Each row is self-contained and easy to export to JSONL for embedding — one record (or one ingredient chunk) per vector.

## Notes

This is a personal project built to explore robust, LLM-assisted data extraction from real-world e-commerce pages. Please scrape responsibly and in line with the target site's terms of service.
