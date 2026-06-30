# Kruidvat RAG

A pipeline that pulls [Kruidvat](https://www.kruidvat.nl)'s cosmetics catalogue into a local SQLite database — product names, marketing descriptions, and clean INCI ingredient lists — then embeds it so you can ask grounded, natural-language questions about it.

The catalogue is a JavaScript-rendered storefront behind bot protection, so the data isn't in the page HTML. Instead of brittle DOM scraping, the pipeline reads clean, structured records straight from Kruidvat's own SAP Commerce (OCC) API — the same API the site uses. The only thing the browser is for is clearing the bot check and cookie consent once; everything after that is fast, structured API calls.

## Motivation

I kept buying cosmetics with ingredients I wanted to avoid. When I asked general AI assistants (ChatGPT, Claude, Gemini) what was in a product, the answers weren't reliable: often based on outdated formulations, confusing the US and EU versions of the same product (which frequently differ), or simply hallucinating ingredients that weren't there.

The problem isn't that the models are bad. A general model's memory is just the wrong place to look for specific, current, region-dependent facts. The fix is grounding: give the model the real, up-to-date ingredient data and let it answer from that instead of from memory.

This project builds that grounding layer: collecting Kruidvat's current EU catalogue into a structured, queryable database of products, their descriptions, and their actual ingredients, ready to ground an LLM for reliable, hallucination-free answers.

## Highlights

- **Structured data, straight from the source**: rather than parsing fragile HTML or running an LLM over each page, it reads clean product records (name, description, INCI ingredients) from Kruidvat's SAP Commerce (OCC) API — fast and reliable, with no per-page extraction step.
- **Gets past the bot wall once**: the catalogue is a JS-rendered SPA behind Akamai bot protection and a OneTrust consent gate. A stealthed Chromium ([`playwright-stealth`](https://pypi.org/project/playwright-stealth/), Chrome's new headless mode, an nl-NL / Europe-Amsterdam profile) clears both once; the API calls then run from inside that browser context and inherit its cookies (a plain server-side request gets a `403`).
- **Captures descriptions, not just ingredients**: each product's use-case/marketing description is saved alongside its INCI list, so questions like *"best shampoo for dry, damaged hair"* work — not only exact-ingredient lookups.
- **Multilingual semantic embeddings**: a second pass embeds each product locally with [`embeddinggemma`](https://ollama.com/library/embeddinggemma) (Google's multilingual embedding model, a good fit for the Dutch descriptions + Latin INCI names) and stores the vectors in the same SQLite file via [sqlite-vec](https://github.com/asg017/sqlite-vec), making the catalogue searchable by meaning.
- **Grounded Q&A**: `query.py` embeds your question, retrieves the closest products, and feeds their descriptions + ingredients to a local LLM (`gemma4`), so answers come from real catalogue data instead of the model's memory.
- **One place to configure**: `config.py` holds the shared defaults (database path, model names, prompt prefixes, Ollama host, and the list of categories to scrape); every script accepts CLI flags that override them per run.
- **Pluggable answer model**: embedding and retrieval are always local, while the model that writes the final answer is selected by `ANSWER_PROVIDER` (local `ollama` today), so a more capable local model or a remote API can be swapped in later without touching retrieval.
- **Idempotent + incremental**: SQLite (WAL mode) with batched inserts, URL de-duplication, and skip-already-scraped logic so runs can be resumed. Embedding is incremental too, so re-runs only process new products.

## Architecture

The three commands and `config.py` live at the repo root; their supporting modules live in `lib/`:

| File | Responsibility |
|------|----------------|
| `config.py` | Shared defaults: database path, model names, prompt prefixes, Ollama host, category list |
| `scraper.py` | Command: scrape a category via the API (list → product details → store) |
| `embed.py` | Command: embed products into a `sqlite-vec` vector table |
| `query.py` | Command: semantic search + grounded Q&A over the catalogue |
| `lib/api.py` | Kruidvat OCC API client: consent/bot-check, paginated product list, product detail |
| `lib/browser.py` | Playwright stealth context + cookie-consent handling |
| `lib/extractor.py` | INCI ingredient-string parsing and cleaning helpers |
| `lib/db.py` | SQLite schema and batched persistence |
| `tests/` | Unit + integration tests (run with `pytest`) |

**Flow:** open the category once (clear consent + bot check, read its `categoryCode`) → page through the OCC **search** API for product URLs → fetch each product's **detail** API for name, description, and ingredients → write to SQLite → embed `name + description + ingredients` with `embeddinggemma` into a `sqlite-vec` table → ask grounded questions with `query.py` (retrieve → answer with `gemma4`).

## Setup

Requires Python 3.9+ and a running [Ollama](https://ollama.com) instance. Ollama is used for **embedding** and **answering** (the scrape itself needs no model), so it must be installed and serving before you embed or query.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv && source .venv/bin/activate

# 2. Install Python dependencies (includes playwright-stealth and sqlite-vec) + the browser
pip install -r requirements.txt
python -m playwright install chromium

# 3. Pull the models and make sure Ollama is running
ollama pull embeddinggemma      # semantic embeddings (embed + query)
ollama pull gemma4:e4b-mlx      # grounded answers (query); use gemma4:e4b off Apple Silicon
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

Override the targets with one or more `--category` flags (repeatable), or cap the size while iterating:

```bash
python scraper.py \
  --category "https://www.kruidvat.nl/verzorging/haarstylingproducten" \
  --limit 10 --db kruidvat.db
```

Useful options:

| Flag | Default | Description |
|------|---------|-------------|
| `--category URL` | `config.CATEGORIES` | Category URL to scrape (repeatable) |
| `--db` | `config.DB_PATH` (`kruidvat.db`) | Output SQLite database |
| `--limit N` | none | Process at most N products per category |
| `--max-pages N` | 200 | Cap on API pages fetched per category |
| `--headed` | off | Show the browser window (debugging) |
| `--proxy URL` | none | Route the browser through a proxy |

A no-argument run scrapes both full categories (~1,000 products) and takes ~15 min, so use `--limit` while iterating.

### 2. Embed for semantic search

After scraping, build a vector for every product so the catalogue can be searched by meaning rather than exact keywords:

```bash
python embed.py --db kruidvat.db
```

This reads products that have ingredients, embeds `name + description + ingredients` with a local Ollama embedding model, and writes the vectors into a `vec_products` table inside the same SQLite file using [sqlite-vec](https://github.com/asg017/sqlite-vec). Re-running only embeds new products, so it is safe to run after each scrape.

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

Once products are embedded, ask questions in natural language. `query.py` embeds your question, retrieves the closest products, and (by default) grounds a local LLM on their descriptions + ingredients:

```bash
# Grounded answer
python query.py "Which shampoo is best for dry and damaged hair?"

# Just retrieve the most relevant products, no LLM
python query.py --search "sulfate-free shampoo"
```

| Flag | Default | Description |
|------|---------|-------------|
| `--db` | `config.DB_PATH` (`kruidvat.db`) | SQLite database to query |
| `--search` | off | Only print retrieved products, skip the LLM answer |
| `--top-k N` | `config.TOP_K` (`10`) | Number of products to retrieve |
| `--provider` | `config.ANSWER_PROVIDER` (`ollama`) | Backend that writes the answer (local for now) |
| `--answer-model` | `config.ANSWER_MODEL` (`gemma4:e4b-mlx`) | Local Ollama model used to write the answer |
| `--embed-model` | `config.EMBED_MODEL` (`embeddinggemma`) | Embedding model (must match `embed.py`) |
| `--ollama-timeout` | `60` | Request timeout (seconds) |

## Tests

Unit tests cover the pure logic with no network, Ollama, or live site needed: ingredient parsing (`lib/extractor.py`), SQLite persistence (`lib/db.py`), the embedding/query helpers, and the answer-provider dispatch. An integration test (`tests/test_vec_integration.py`) runs a full embed → search through real `sqlite-vec` using deterministic fake embeddings, so it verifies the vector wiring and the KNN query without a model server. A live end-to-end test (`tests/test_e2e_ollama.py`) runs the real embed → answer path and auto-skips unless Ollama and the models are present.

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
  description     TEXT,    -- product description from the API
  ingredients_list TEXT,   -- JSON array of normalized INCI ingredients
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

## To do

- **Hybrid keyword + vector search.** Semantic search is great for use-case questions ("good for dry hair") but vague for exact-ingredient ones ("everything without SLS", "which contain Limonene"). A SQL `LIKE` / keyword pre-filter on `ingredients_list`, combined with the vector results, would make ingredient-exact questions exhaustive. No new dependency — plain SQLite.
- **Test and compare different embedding models.** `embeddinggemma` is the current default: multilingual and a 768-dim drop-in. Worth benchmarking on real queries against `bge-m3` (1024-dim; strong multilingual, needs `EMBED_DIM = 1024` + a `--reset` re-embed), `qwen3-embedding:0.6b`, and `nomic-embed-text` (English baseline). Remember the per-model prompt prefixes and re-embed with `--reset` when switching.
- **Set up an evaluation harness** so the comparison above is repeatable rather than eyeballed.
- **Add a remote answer provider** (e.g. Anthropic / OpenAI) as a branch in `query.py`'s `generate_answer()`, selected via `ANSWER_PROVIDER`, with the API key read from the environment. Embedding and retrieval stay local.
- **Capture more product attributes.** The OCC product API also exposes hair-type and other classification fields; saving those could sharpen use-case queries further (note: hair *porosity* is not in the data).

## Notes

This is a personal project for grounding a local LLM on real, current catalogue data. The product data comes from Kruidvat's own API; please use it responsibly and in line with the site's terms of service.
