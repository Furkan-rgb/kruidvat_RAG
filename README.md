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
- **Multilingual semantic embeddings**: a second pass embeds each product locally with [`embeddinggemma`](https://ollama.com/library/embeddinggemma) (Google's multilingual embedding model, a good fit for the Dutch product text + Latin INCI names) and stores the vectors in the same SQLite file via [sqlite-vec](https://github.com/asg017/sqlite-vec), making the catalogue searchable by meaning.
- **Grounded Q&A**: `query.py` embeds your question, retrieves the closest products, and feeds them to a local LLM (`gemma4`) as context, so answers come from real catalogue data instead of the model's memory.
- **One place to configure**: `config.py` holds the shared defaults (database path, model names, prompt prefixes, Ollama host, and the list of categories to scrape); every script accepts CLI flags that override them per run.
- **Pluggable answer model**: embedding and retrieval are always local, while the model that writes the final answer is selected by `ANSWER_PROVIDER` (local `ollama` today), so a more capable local model or a remote API can be swapped in later without touching retrieval.
- **Smart pagination**: reads the total product count from the category page and computes exactly how many pages to crawl, with empty-page short-circuiting.
- **Idempotent storage**: SQLite (WAL mode) with batched inserts, URL de-duplication, and skip-already-scraped logic so runs can be resumed. Embedding is incremental too, so re-runs only process new products.

## Architecture

The code is split into focused, independently testable modules:

| File | Responsibility |
|------|----------------|
| `config.py` | Shared defaults: database path, model names, prompt prefixes, Ollama host, category list |
| `scraper.py` | CLI entry point and orchestration (crawl → extract → store) |
| `browser.py` | Playwright context/stealth setup, navigation, link & name extraction |
| `pager.py` | Category pagination and product-link collection |
| `extractor.py` | HTML sanitization + LLM ingredient extraction and response parsing |
| `db.py` | SQLite schema and batched persistence |
| `embed.py` | Embeds products into a `sqlite-vec` vector table for semantic search |
| `query.py` | Semantic search + grounded Q&A over the embedded catalogue |
| `logging_config.py` | Structured JSON logging |
| `tests/` | Unit + integration tests (run with `pytest`) |

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
ollama pull ministral-3:3b      # ingredient extraction (scraper)
ollama pull embeddinggemma      # semantic embeddings (embed + query)
ollama pull gemma4:e4b          # grounded answers (query)
ollama serve   # if not already running (default: http://localhost:11434)
```

## Configuration

Shared defaults live in `config.py`: the database path, model names, the Ollama host, and the `CATEGORIES` list that `scraper.py` crawls by default. Edit that file to change them globally, or pass the matching CLI flag to override a single run.

Two things worth knowing:

- **Embedding prompt prefixes.** EmbeddingGemma embeds documents and queries with different task instructions, kept in `config.py` as `EMBED_DOC_PREFIX` / `EMBED_QUERY_PREFIX`. If you switch to a model with other conventions (e.g. `bge-m3` needs none), update or empty them there.
- **Pluggable answer model.** The model that writes the final answer is selected by `ANSWER_PROVIDER`, which is `ollama` (local) for now. Embedding and retrieval are always local; a remote provider can be added later as a single branch in `query.py`'s `generate_answer()` without touching anything else.

## Usage

The full pipeline is three steps: scrape, embed, then query.

### 1. Scrape

With no arguments, this scrapes every category in `config.CATEGORIES`:

```bash
python scraper.py
```

Override the targets with one or more `--category` flags (repeatable):

```bash
python scraper.py \
  --category "https://www.kruidvat.nl/verzorging/haarstylingproducten" \
  --db kruidvat.db
```

Useful options:

| Flag | Default | Description |
|------|---------|-------------|
| `--category URL` | `config.CATEGORIES` | Category URL to scrape (repeatable) |
| `--db` | `config.DB_PATH` (`kruidvat.db`) | Output SQLite database |
| `--headed` | off | Show the browser window (debugging) |
| `--concurrency N` | 10 | Concurrent product pages |
| `--max-pages N` | 200 | Cap on category pages crawled |
| `--limit N` | none | Process at most N products per category |
| `--delay S` | 0.2 | Delay between paginated requests |
| `--ollama-model` | `config.EXTRACT_MODEL` | Local Ollama model used for extraction |
| `--ollama-url` | `config.GENERATE_URL` | Ollama generate endpoint |
| `--ollama-timeout` | `60` | LLM request timeout (seconds) |

### 2. Embed for semantic search

After scraping, build a vector for every product so the catalogue can be searched by meaning rather than exact keywords:

```bash
python embed.py --db kruidvat.db
```

This reads products that have ingredients, embeds `name + ingredients` with a local Ollama embedding model, and writes the vectors into a `vec_products` table inside the same SQLite file using [sqlite-vec](https://github.com/asg017/sqlite-vec). Re-running only embeds new products, so it is safe to run after each scrape.

**Changing the embedding model?** Re-run with `--reset`. The incremental skip can't tell the model changed, and vectors from different models are not comparable, so `--reset` drops and rebuilds the vector table first.

| Flag | Default | Description |
|------|---------|-------------|
| `--db` | `config.DB_PATH` (`kruidvat.db`) | SQLite database written by `scraper.py` |
| `--embed-model` | `config.EMBED_MODEL` (`embeddinggemma`) | Local Ollama embedding model |
| `--embed-dim` | `config.EMBED_DIM` (`768`) | Embedding size (must match the model's output) |
| `--ollama-url` | `config.EMBEDDINGS_URL` | Ollama embeddings endpoint |
| `--ollama-timeout` | `60` | Embedding request timeout (seconds) |
| `--reset` | off | Drop and rebuild the vector table first (use when changing the embedding model) |

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
| `--db` | `config.DB_PATH` (`kruidvat.db`) | SQLite database to query |
| `--search` | off | Only print retrieved products, skip the LLM answer |
| `--top-k N` | `config.TOP_K` (`5`) | Number of products to retrieve |
| `--provider` | `config.ANSWER_PROVIDER` (`ollama`) | Backend that writes the answer (local for now) |
| `--answer-model` | `config.ANSWER_MODEL` (`gemma4:e4b`) | Local Ollama model used to write the answer |
| `--embed-model` | `config.EMBED_MODEL` (`embeddinggemma`) | Embedding model (must match `embed.py`) |
| `--ollama-timeout` | `60` | Request timeout (seconds) |

## Tests

Unit tests cover the pure logic with no network, Ollama, or live site needed: ingredient parsing (`extractor.py`), SQLite persistence (`db.py`), the embedding/query helpers, and the answer-provider dispatch. An integration test (`tests/test_vec_integration.py`) runs a full embed → search through real `sqlite-vec` using deterministic fake embeddings, so it verifies the vector wiring and the KNN query without a model server.

```bash
pip install -r requirements.txt       # runtime deps (beautifulsoup4, sqlite-vec, ...)
pip install -r requirements-dev.txt   # pytest
pytest
```

The integration test skips automatically if the `sqlite-vec` extension can't load in your Python build (some system Python builds disable `enable_load_extension`).

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
  embedding   FLOAT[768]            -- embeddinggemma vector (768-dim)
)
```

Each row is self-contained and easy to export to JSONL for embedding, one record (or one ingredient chunk) per vector.

## To do

- **Test and compare different embedding models.** `embeddinggemma` is the current default: multilingual (a good fit for the Dutch text + Latin INCI names), pairs with the Gemma 4 answer model, and is a 768-dim drop-in. It's still worth benchmarking on real queries against:
  - `embeddinggemma` (768-dim, current default)
  - `bge-m3` (1024-dim; strongest multilingual retrieval, needs `EMBED_DIM = 1024` and a `--reset` re-embed)
  - `qwen3-embedding:0.6b` (flexible dims; top sub-1GB multilingual)
  - `nomic-embed-text` (English-focused baseline)

  Both `embed.py` and `query.py` take `--embed-model` / `--embed-dim` (or set `EMBED_MODEL` / `EMBED_DIM` in `config.py`); remember the per-model prompt prefixes (`EMBED_DOC_PREFIX` / `EMBED_QUERY_PREFIX`) and re-embed with `--reset` when switching. Pick a set of representative questions, measure retrieval quality (and answer quality) per model, and record the results here.
- **Set up an evaluation harness** so the comparison above is repeatable rather than eyeballed.
- **Add a remote answer provider** (e.g. Anthropic / OpenAI) as a branch in `query.py`'s `generate_answer()`, selected via `ANSWER_PROVIDER`, with the API key read from the environment. Embedding and retrieval stay local.

## Notes

This is a personal project built to explore robust, LLM-assisted data extraction from real-world e-commerce pages. Please scrape responsibly and in line with the target site's terms of service.
