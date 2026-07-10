"""Pydantic request and response contracts for the HTTP API."""

from typing import Literal

from pydantic import BaseModel, Field, constr

import config
from app.service import MAX_TOP_K


class AskRequest(BaseModel):
    question: constr(strip_whitespace=True, min_length=1)  # type: ignore[valid-type]
    mode: Literal["advisor", "strict"] = config.ANSWER_MODE
    top_k: int = Field(default=config.TOP_K, ge=1, le=MAX_TOP_K)


class ModelInfo(BaseModel):
    embedding: str
    answer: str
    provider: str
    embedding_provider: str | None = None


class ProductResultResponse(BaseModel):
    rank: int
    distance: float
    id: int
    name: str | None
    url: str | None
    description: str | None
    ingredients: list[str]


class AskResponse(BaseModel):
    question: str
    mode: Literal["advisor", "strict"]
    top_k: int
    answer: str
    products: list[ProductResultResponse]
    models: ModelInfo
    elapsed_ms: float


class StoredProductResponse(BaseModel):
    id: int
    name: str | None
    url: str | None
    description: str | None
    ingredients: list[str]
    scraped_at: str | None


class HealthResponse(BaseModel):
    status: str
    database_exists: bool
    products_table_exists: bool
    product_count: int | None
    vec_products_table_exists: bool
    embedded_product_count: int | None
    models: ModelInfo
