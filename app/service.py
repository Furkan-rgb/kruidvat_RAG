"""Shared semantic retrieval and grounded-answer service.

The CLI and HTTP API both use this module.  It deliberately keeps the
original one-embedding -> one-KNN-search -> one-answer-call design.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import sqlite_vec
from sqlite_vec import serialize_float32

import config
from app.index_metadata import profile_mismatches, provider_profile, read_index_metadata
from app.providers import (
    AnswerProvider,
    EmbeddingProvider,
    OllamaAnswerProvider,
    OllamaEmbeddingProvider,
    ProviderError,
    create_answer_provider,
    create_embedding_provider,
)
from app.providers.ollama import post_json as _provider_post_json
from app.providers.ollama import translate_error as _provider_translate_ollama_error

MAX_TOP_K = 25
VALID_MODES = {"advisor", "strict"}

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

SYSTEM_PROMPTS = {"advisor": ADVISOR_SYSTEM_PROMPT, "strict": GROUNDED_SYSTEM_PROMPT}
CLOSING_INSTRUCTIONS = {
    "advisor": (
        "Recommend from the products above. Combine their ingredients and "
        "descriptions with general haircare knowledge, and keep clear what is "
        "catalogue fact and what is your own judgement."
    ),
    "strict": "Answer using only the context above.",
}


class ServiceError(Exception):
    """A known failure that can be safely presented to an API client."""

    def __init__(self, code: str, message: str, remediation: str):
        super().__init__(message)
        self.code = code
        self.message = message
        self.remediation = remediation

    def detail(self) -> dict[str, str]:
        return {
            "code": self.code,
            "message": self.message,
            "remediation": self.remediation,
        }


class InputError(ValueError):
    """Invalid service input (normally caught by the HTTP schema first)."""


@dataclass(frozen=True)
class ProductResult:
    rank: int
    distance: float
    id: int
    name: str | None
    url: str | None
    description: str | None
    ingredients: list[str]


@dataclass(frozen=True)
class StoredProduct:
    id: int
    name: str | None
    url: str | None
    description: str | None
    ingredients: list[str]
    scraped_at: str | None


@dataclass(frozen=True)
class QueryResult:
    question: str
    mode: str
    top_k: int
    answer: str
    products: list[ProductResult]
    models: dict[str, str]
    elapsed_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _post_json(url, payload, timeout):
    """POST a JSON payload to a local Ollama endpoint and return the parsed body."""
    return _provider_post_json(url, payload, timeout)


def get_embedding(text, *, model, url, timeout, _post=None):
    """Embed `text` with the local Ollama embedding model."""
    adapter = OllamaEmbeddingProvider(
        model=model,
        dimension=0,
        url=url,
        timeout=timeout,
        _post=_post or _post_json,
    )
    return adapter.embed_query(text)


def ingredients_to_text(ingredients_list):
    """Convert a stored JSON ingredient array into prompt-friendly text."""
    try:
        items = json.loads(ingredients_list)
        if isinstance(items, list):
            return ", ".join(str(i) for i in items)
    except Exception:
        pass
    return ingredients_list or ""


def parse_ingredients(ingredients_list) -> list[str]:
    """Return the structured ingredient array; malformed stored data is empty."""
    try:
        items = json.loads(ingredients_list)
    except (TypeError, ValueError):
        return []
    if not isinstance(items, list):
        return []
    return [str(item).strip() for item in items if str(item).strip()]


def search(conn, query_vector, top_k):
    """Return nearest products in ascending sqlite-vec distance order."""
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
    """Build the original product-scoped grounded context block."""
    blocks = []
    for i, (_dist, _pid, name, url, description, ingredients_list) in enumerate(rows, 1):
        block = f"Product {i}: {name}\nURL: {url}\n"
        if description:
            block += f"Description: {description}\n"
        block += f"Ingredients of {name}: {ingredients_to_text(ingredients_list)}"
        blocks.append(block)
    return "\n\n----------\n\n".join(blocks)


def _answer_ollama(prompt, *, model, url, timeout, system, _post=None):
    return OllamaAnswerProvider(
        model=model, url=url, timeout=timeout, _post=_post or _post_json
    ).generate(system, prompt)


def _stream_answer_ollama(prompt, *, model, url, timeout, system):
    """Yield answer text chunks from Ollama's newline-delimited JSON stream."""
    yield from OllamaAnswerProvider(model=model, url=url, timeout=timeout).stream(
        system, prompt
    )


def build_answer_prompt(question: str, context: str, mode: str) -> tuple[str, str]:
    """Return the unchanged mode-specific system and user prompts."""
    system = SYSTEM_PROMPTS[mode]
    prompt = (
        f"Context (retrieved products):\n{context}\n\n"
        f"Question: {question}\n\n"
        f"{CLOSING_INSTRUCTIONS[mode]}"
    )
    return system, prompt


def generate_answer(
    question, context, *, provider, model, url, timeout, mode="advisor", _post=None
):
    """Write the final answer using the original mode-specific prompts."""
    system, prompt = build_answer_prompt(question, context, mode)
    if provider == "ollama" and _post is not None:
        adapter: AnswerProvider = OllamaAnswerProvider(
            model=model, url=url, timeout=timeout, _post=_post
        )
    else:
        adapter = create_answer_provider(
            provider, model=model, url=url, timeout=timeout
        )
    return adapter.generate(system, prompt)


def rows_to_products(rows) -> list[ProductResult]:
    return [
        ProductResult(
            rank=rank,
            distance=float(row[0]),
            id=int(row[1]),
            name=row[2],
            url=row[3],
            description=row[4],
            ingredients=parse_ingredients(row[5]),
        )
        for rank, row in enumerate(rows, 1)
    ]


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone() is not None


def _translate_ollama_error(exc: Exception, model: str) -> ServiceError:
    provider_error = _provider_translate_ollama_error(exc, model)
    return ServiceError(
        provider_error.code, provider_error.message, provider_error.remediation
    )


def _translate_provider_error(exc: ProviderError) -> ServiceError:
    return ServiceError(exc.code, exc.message, exc.remediation)


class RAGService:
    """Configurable shared application service for CLI and HTTP callers."""

    def __init__(
        self,
        *,
        db_path: str = config.DB_PATH,
        embed_model: str = config.EMBED_MODEL,
        embed_dim: int = config.EMBED_DIM,
        answer_model: str = config.ANSWER_MODEL,
        provider: str = config.ANSWER_PROVIDER,
        embed_provider: str = config.EMBED_PROVIDER,
        embed_url: str = config.EMBEDDINGS_URL,
        generate_url: str = config.GENERATE_URL,
        timeout: float = config.OLLAMA_TIMEOUT,
        embedding_provider: EmbeddingProvider | None = None,
        answer_provider: AnswerProvider | None = None,
    ):
        self.db_path = db_path
        self.embed_model = embed_model
        self.answer_model = answer_model
        self.provider = provider
        self.embed_provider = embed_provider
        self.embed_url = embed_url
        self.generate_url = generate_url
        self.timeout = timeout
        self.embedding_provider = embedding_provider or create_embedding_provider(
            embed_provider,
            model=embed_model,
            dimension=embed_dim,
            url=embed_url,
            timeout=timeout,
            document_prefix=config.EMBED_DOC_PREFIX,
            query_prefix=config.EMBED_QUERY_PREFIX,
        )
        self.answer_provider = answer_provider or create_answer_provider(
            provider, model=answer_model, url=generate_url, timeout=timeout
        )

        # Provider objects are authoritative when explicitly injected.
        self.embed_model = self.embedding_provider.model
        self.answer_model = self.answer_provider.model
        self.embed_provider = self.embedding_provider.provider
        self.provider = self.answer_provider.provider

    @property
    def models(self) -> dict[str, str]:
        return {
            "embedding": self.embed_model,
            "answer": self.answer_model,
            "provider": self.provider,
            "embedding_provider": self.embed_provider,
        }

    def _connect(self, *, require_vector: bool) -> sqlite3.Connection:
        if not Path(self.db_path).is_file():
            raise ServiceError(
                "database_missing",
                f"The configured database {self.db_path!r} does not exist.",
                "Run `python scraper.py` to create and populate it.",
            )
        try:
            conn = sqlite3.connect(self.db_path)
        except sqlite3.Error as exc:
            raise ServiceError(
                "database_unavailable",
                "The catalogue database could not be opened.",
                "Check DB_PATH in config.py and the database file permissions.",
            ) from exc
        if not _table_exists(conn, "products"):
            conn.close()
            raise ServiceError(
                "products_table_missing",
                "The products table is missing from the catalogue database.",
                "Run `python scraper.py` to initialize and populate the database.",
            )
        if not require_vector:
            return conn
        if not _table_exists(conn, "vec_products"):
            conn.close()
            raise ServiceError(
                "vector_index_missing",
                "The vector index is missing from the catalogue database.",
                "Run `python embed.py` after scraping products.",
            )
        try:
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
        except Exception as exc:
            conn.close()
            raise ServiceError(
                "sqlite_vec_unavailable",
                "The sqlite-vec extension could not be loaded.",
                "Reinstall sqlite-vec and use a Python build that permits SQLite extensions.",
            ) from exc
        stored_profile = read_index_metadata(conn)
        if stored_profile is not None:
            mismatches = profile_mismatches(
                stored_profile, provider_profile(self.embedding_provider)
            )
            if mismatches:
                conn.close()
                raise ServiceError(
                    "embedding_index_mismatch",
                    "The configured embedding provider does not match the stored vector index "
                    f"({', '.join(mismatches)} differ).",
                    "Run `python embed.py --reset` with the configured embedding provider.",
                )
        return conn

    @staticmethod
    def validate(question: str, mode: str, top_k: int) -> str:
        question = question.strip()
        if not question:
            raise InputError("Question must not be blank.")
        if mode not in VALID_MODES:
            raise InputError("Mode must be 'advisor' or 'strict'.")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= MAX_TOP_K:
            raise InputError(f"top_k must be between 1 and {MAX_TOP_K}.")
        return question

    def _embed_question(self, question: str):
        try:
            return self.embedding_provider.embed_query(question)
        except ProviderError as exc:
            raise _translate_provider_error(exc) from exc

    @staticmethod
    def _search_rows(conn: sqlite3.Connection, query_vector, top_k: int):
        try:
            return search(conn, query_vector, top_k)
        except sqlite3.Error as exc:
            raise ServiceError(
                "vector_index_unavailable",
                "The vector index could not be queried.",
                "Run `python embed.py --reset` to rebuild the vector index.",
            ) from exc

    def retrieve(self, question: str, *, mode: str = "advisor", top_k: int = config.TOP_K):
        question = self.validate(question, mode, top_k)
        conn = self._connect(require_vector=True)
        try:
            query_vector = self._embed_question(question)
            rows = self._search_rows(conn, query_vector, top_k)
            return question, rows
        finally:
            conn.close()

    def ask(self, question: str, *, mode: str = "advisor", top_k: int = config.TOP_K) -> QueryResult:
        started = time.perf_counter()
        question, rows = self.retrieve(question, mode=mode, top_k=top_k)
        products = rows_to_products(rows)
        if not rows:
            answer = "No matching products were retrieved. Run the embedding step if the catalogue should contain products."
        else:
            try:
                system, prompt = build_answer_prompt(question, build_context(rows), mode)
                answer = self.answer_provider.generate(system, prompt)
            except ServiceError:
                raise
            except ProviderError as exc:
                raise _translate_provider_error(exc) from exc
        return QueryResult(
            question=question,
            mode=mode,
            top_k=top_k,
            answer=answer,
            products=products,
            models=self.models,
            elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    def stream_ask(
        self, question: str, *, mode: str = "advisor", top_k: int = config.TOP_K
    ):
        """Yield structured progress, evidence, answer-token, and completion events."""
        started = time.perf_counter()
        question = self.validate(question, mode, top_k)
        yield {
            "type": "status",
            "stage": "embedding",
            "message": "Embedding the complete question...",
        }

        conn = self._connect(require_vector=True)
        try:
            query_vector = self._embed_question(question)
            yield {
                "type": "status",
                "stage": "retrieving",
                "message": "Searching the catalogue for the closest products...",
            }
            rows = self._search_rows(conn, query_vector, top_k)
        finally:
            conn.close()

        products = rows_to_products(rows)
        yield {
            "type": "evidence",
            "question": question,
            "mode": mode,
            "top_k": top_k,
            "products": [asdict(product) for product in products],
            "models": self.models,
        }

        if not rows:
            answer = (
                "No matching products were retrieved. Run the embedding step if "
                "the catalogue should contain products."
            )
            yield {"type": "token", "text": answer}
            yield {
                "type": "done",
                "answer": answer,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            }
            return

        yield {
            "type": "status",
            "stage": "generating",
            "message": "Generating a grounded answer from the retrieved evidence...",
        }
        system, prompt = build_answer_prompt(question, build_context(rows), mode)
        chunks: list[str] = []
        try:
            for chunk in self.answer_provider.stream(system, prompt):
                chunks.append(chunk)
                yield {"type": "token", "text": chunk}
        except ProviderError as exc:
            raise _translate_provider_error(exc) from exc

        yield {
            "type": "done",
            "answer": "".join(chunks).strip(),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    def get_product(self, product_id: int) -> StoredProduct | None:
        conn = self._connect(require_vector=False)
        try:
            row = conn.execute(
                "SELECT id, name, url, description, ingredients_list, scraped_at "
                "FROM products WHERE id=?",
                (product_id,),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            return None
        return StoredProduct(
            id=int(row[0]),
            name=row[1],
            url=row[2],
            description=row[3],
            ingredients=parse_ingredients(row[4]),
            scraped_at=row[5],
        )

    def health(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "status": "database_missing",
            "database_exists": Path(self.db_path).is_file(),
            "products_table_exists": False,
            "product_count": None,
            "vec_products_table_exists": False,
            "embedded_product_count": None,
            "models": self.models,
        }
        if not result["database_exists"]:
            return result
        try:
            conn = sqlite3.connect(self.db_path)
        except sqlite3.Error:
            result["status"] = "database_unavailable"
            return result
        try:
            result["products_table_exists"] = _table_exists(conn, "products")
            if not result["products_table_exists"]:
                result["status"] = "products_table_missing"
                return result
            result["product_count"] = conn.execute("SELECT COUNT(*) FROM products").fetchone()[0]
            result["vec_products_table_exists"] = _table_exists(conn, "vec_products")
            if not result["vec_products_table_exists"]:
                result["status"] = "vector_index_missing"
                return result
            try:
                conn.enable_load_extension(True)
                sqlite_vec.load(conn)
                conn.enable_load_extension(False)
                result["embedded_product_count"] = conn.execute(
                    "SELECT COUNT(*) FROM vec_products"
                ).fetchone()[0]
            except Exception:
                result["status"] = "sqlite_vec_unavailable"
                return result
            stored_profile = read_index_metadata(conn)
            if stored_profile is not None and profile_mismatches(
                stored_profile, provider_profile(self.embedding_provider)
            ):
                result["status"] = "embedding_index_mismatch"
                return result
            result["status"] = "ready"
            return result
        finally:
            conn.close()
