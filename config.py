"""config.py: shared defaults for the scrape / embed / query pipeline.

These are the defaults used across scraper.py, embed.py and query.py. Every
script still exposes CLI flags that override anything here on a per-run basis,
so this file is the single place to change the database name, the local model
choices, the Ollama host, or the list of categories to scrape.
"""

# Where the scraped catalogue and its embeddings are stored.
DB_PATH = "kruidvat.db"

# Local Ollama server. Extraction and answering use /api/generate; embedding
# uses /api/embeddings.
OLLAMA_HOST = "http://localhost:11434"
GENERATE_URL = f"{OLLAMA_HOST}/api/generate"
EMBEDDINGS_URL = f"{OLLAMA_HOST}/api/embeddings"
OLLAMA_TIMEOUT = 60.0  # seconds, per request

# Models (pull each with `ollama pull <name>`).
EXTRACT_MODEL = "ministral-3:3b"   # ingredient extraction (scraper.py)
ANSWER_MODEL = "ministral-3:3b"    # grounded answers (query.py)
EMBED_MODEL = "nomic-embed-text"   # semantic embeddings (embed.py / query.py)
EMBED_DIM = 768                    # must match EMBED_MODEL's output size

# Retrieval.
TOP_K = 5  # products retrieved per query (query.py)

# Categories scraped by default when scraper.py is run with no --category flag.
CATEGORIES = [
    "https://www.kruidvat.nl/verzorging/haarstylingproducten",
    "https://www.kruidvat.nl/verzorging/haarverzorging",
]
