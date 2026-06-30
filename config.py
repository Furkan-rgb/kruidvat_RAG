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

# Models (pull each with `ollama pull <name>`). The "-mlx" tag uses Ollama's
# MLX engine and is Apple-Silicon only; on other platforms use the portable
# GGUF tag instead (e.g. "gemma4:e4b").
EMBED_MODEL = "embeddinggemma"  # semantic embeddings (embed.py / query.py)
EMBED_DIM = 768  # must match EMBED_MODEL's output size

# EmbeddingGemma embeds documents and queries with different task instructions.
# These prefixes travel with EMBED_MODEL, so change (or empty) them if you switch
# to a model with other conventions: bge-m3 needs none; nomic-embed-text uses
# "search_document: " / "search_query: ".
EMBED_DOC_PREFIX = "title: none | text: "  # product text (embed.py)
EMBED_QUERY_PREFIX = "task: search result | query: "  # the question (query.py)

# Answer step: the "main model" that writes the final grounded answer (query.py).
# Embedding and retrieval are always local; only this step's backend is
# pluggable. A remote provider can be added later as another branch in
# query.py's generate_answer() without touching anything else.
ANSWER_PROVIDER = "ollama"  # implemented: "ollama" (local Ollama)
ANSWER_MODEL = "gemma4:e4b-mlx"  # for ollama, any local chat model

# Retrieval.
TOP_K = 10  # products retrieved per query (query.py)

# Categories scraped by default when scraper.py is run with no --category flag.
CATEGORIES = [
    "https://www.kruidvat.nl/verzorging/haarstylingproducten",
    "https://www.kruidvat.nl/verzorging/haarverzorging",
]
