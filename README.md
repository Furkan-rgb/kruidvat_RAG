# Kruidvat Ingredient Scraper

A web scraping pipeline that crawls product pages from [Kruidvat](https://www.kruidvat.nl) and extracts clean, structured cosmetic ingredient data (INCI) into a local SQLite database, then embeds it so you can ask grounded, natural-language questions about the catalogue.

The interesting part isn't the crawling; it's getting *reliable structured data* out of messy, inconsistent webshop HTML. Instead of brittle CSS-selector parsing, a local LLM reads the sanitized page text and returns a validated ingredient list, while distinguishing real cosmetics from hardware (hair dryers, brushes, etc.) that have no ingredients at all.

## Motivation

I kept buying cosmetics with ingredients I wanted to avoid. When I asked general AI assistants (ChatGPT, Claude, Gemini) what was in a product, the answers weren't reliable: often based on outdated formulations, confusing the US and EU versions of the same product (which frequently differ), or simply hallucinating ingredients that weren't there.

The problem isn't that the models are bad. A general model's memory is just the wrong place to look for specific, current, region-dependent facts. The fix is grounding: give the model the real, up-to-date ingredient data and let it answer from that instead of from memory.

This project builds that grounding layer: scraping Kruidvat's current EU catalogue into a structured, queryable database of products and their actual ingredients, ready to ground an LLM for reliable, hallucination-free answers.

## Highlights

- **Async + concurrent**: built on `asyncio` and Playwright; product pages are scraped in parallel with a bounded semaphore.
- **Anti-bot hardening**: uses [`playwright-stealth`](https://pypi.org/project/playwright-stealth/) to patch the automation fingerprint, plus a realistic Chromium profile (nl-NL locale, Europe/Amsterdam timezone, custom user-agent/headers, `--disable-blink-features=AutomationControlled`) and automatic cookie/"read more" handling.
- **Local LLM extraction via Ollama**: each product page's sanitized text is sent to a local [Ollama](https://ollama.com) model over its HTTP API (`/api/generate`) with a schema-constrained, INCI-focused prompt (`{"found": bool, "ingredients": [...]}`), `temperature: 0`, and `format: json`. Calls run off the event loop and are serialized through a single-flight semaphore. No cloud APIs or keys; everything runs on your machine.
- **Semantic embeddings**: a second pass embeds each product locally with [`nomic-embed-text`](https://ollama.com/library/nomic-embed-text) and stores the vectors in the same SQLite file via [sqlite-vec](https://github.com/asg017/sqlite-vec), making the catalogue searchable by meaning.
- **Grounded Q&A**: `query.py` embeds your question, retrieves the closest products, and feeds them to a local LLM as context, so answers come from real catalogue data instead of the model's memory.
- **Smart pagination**: reads the total product count from the category page and computes exactly how many pages to crawl, with empty-page short-circuiting.
- **Idempotent storage**: SQLite (WAL mode) with batched inserts, URL de-duplication, and skip-already-scraped logic so runs can be resumed. Embedding is incremental too, so re-runs only process new products.

## Architecture

The code is split into focused, independently testable modules:

| File | Responsibility |
|------|----------------|
| `scraper.py` | CLI entry point and orchestration (crawl → extract → store) |
| `browser.py` | Playwright context/stealth setup, navigation, link & name extraction |
| `pager.py` | Category pagination and product-link collection |
| `extractor.py` | HTML sanitization + LLM ingredient extraction and response parsing |
| `db.py` | SQLite schema and batched persistence |
| `embed.py` | Embeds products into a `sqlite-vec` vector table for semantic search |
| `query.py` | Semantic search + grounded Q&A over the embedded catalogue |
| `logging_config.py` | Structured JSON logging |

**Flow:** open category page → read total product count → paginate and collect `/p/` product links → filter out already-scraped URLs → concurrently visit each product, sanitize the HTML, and ask the local LLM for ingredients → batch-write results to SQLite → embed each product into a vector table → ask grounded natural-language questions with `query.py`.

## Setup

Requires Python 3.9+ and a running [Ollama](https://ollama.com) instance; extraction, embedding, and answering all call Ollama locally, so it must be installed and serving before you run them.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv && source .venv/bin/activate

# 2. Install Python dependencies (includes playwright-stealth and sqlite-vec) + the browser
pip install -r requirements.txt
python -m playwright install chromium

# 3. Pull the models and make sure Ollama is running
ollama pull ministral-3:3b      # ingredient extraction + grounded answers
ollama pull nomic-embed-text    # semantic embeddings
ollama serve   # if not already running (default: http://localhost:11434)
```

## Usage

The full pipeline is three steps: scrape, embed, then query. `start.sh` runs the first two for a couple of categories.

### 1. Scrape

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
| `--limit N` | none | Process at most N products |
| `--delay S` | 0.2 | Delay between paginated requests |
| `--ollama-model` | `ministral-3:3b` | Local Ollama model used for extraction |
| `--ollama-url` | `http://localhost:11434/api/generate` | Ollama generate endpoint |
| `--ollama-timeout` | `60` | LLM request timeout (seconds) |

### 2. Embed for semantic search

After scraping, build a vector for every product so the catalogue can be searched by meaning rather than exact keywords:

```bash
python embed.py --db kruidvat.db
```

This reads products that have ingredients, embeds `name + ingredients` with a local Ollama embedding model, and writes the vectors into a `vec_products` table inside the same SQLite file using [sqlite-vec](https://github.com/asg017/sqlite-vec). Re-running only embeds new products, so it is safe to run after each scrape.

| Flag | Default | Description |
|------|---------|-------------|
| `--db` | `kruidvat.db` | SQLite database written by `scraper.py` |
| `--embed-model` | `nomic-embed-text` | Local Ollama embedding model |
| `--embed-dim` | `768` | Embedding size (must match the model's output) |
| `--ollama-url` | `http://localhost:11434/api/embeddings` | Ollama embeddings endpoint |
| `--ollama-timeout` | `60` | Embedding request timeout (seconds) |

### 3. Ask questions (RAG)

Once products are embedded, ask questions in natural language. `query.py` embeds your question, retrieves the closest products, and (by default) grounds a local LLM on them so the answer comes from real ingredient data:

```bash
# Grounded answer
python query.py "Which hairsprays are alcohol-free?"

# Just retrieve the most relevant products, no LLM
python query.py --search "contains Linalool"
```

| Flag | Default | Description |
|------|---------|-------------|
| `--db` | `kruidvat.db` | SQLite database to query |
| `--search` | off | Only print retrieved products, skip the LLM answer |
| `--top-k N` | 5 | Number of products to retrieve |
| `--answer-model` | `ministral-3:3b` | Local Ollama model used to write the answer |
| `--embed-model` | `nomic-embed-text` | Embedding model (must match `embed.py`) |
| `--ollama-timeout` | `60` | Request timeout (seconds) |

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

`embed.py` adds a companion virtual table in the same file:

```sql
vec_products USING vec0(
  product_id  INTEGER PRIMARY KEY,  -- references products.id
  embedding   FLOAT[768]            -- nomic-embed-text vector
)
```

Each row is self-contained and easy to export to JSONL for embedding, one record (or one ingredient chunk) per vector.

## To do

- **Test and compare different embedding models.** `nomic-embed-text` is the current default mostly because it runs locally with a single `ollama pull` and has a good quality-to-size ratio; it was not chosen by benchmarking on this data. The catalogue is an unusual mix of Dutch marketing text and Latin INCI ingredient names, so the right embedder is an open question. Worth evaluating on real queries:
  - `nomic-embed-text` (768-dim, current default)
  - `bge-m3` and `multilingual-e5` (multilingual; likely stronger on the Dutch text)
  - `mxbai-embed-large` (1024-dim; higher general English scores, but heavier)
  - `all-minilm` (tiny and fast; useful as a cheap baseline)

  Both `embed.py` and `query.py` already take `--embed-model` / `--embed-dim`, so swapping models is just a flag plus a re-embed. Pick a set of representative questions, measure retrieval quality (and answer quality) per model, and record the results here.
- **Set up an evaluation harness** so the comparison above is repeatable rather than eyeballed.

## Notes

This is a personal project built to explore robust, LLM-assisted data extraction from real-world e-commerce pages. Please scrape responsibly and in line with the target site's terms of service.
