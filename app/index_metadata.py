"""Embedding-index identity stored alongside sqlite-vec vectors."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any

METADATA_TABLE = "embedding_index_metadata"
INDEX_NAME = "vec_products"


def ensure_metadata_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {METADATA_TABLE} (
            index_name TEXT PRIMARY KEY,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            dimension INTEGER NOT NULL,
            document_prefix TEXT NOT NULL,
            query_prefix TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )


def read_index_metadata(conn: sqlite3.Connection) -> dict[str, Any] | None:
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (METADATA_TABLE,),
    ).fetchone()
    if exists is None:
        return None
    row = conn.execute(
        f"SELECT provider, model, dimension, document_prefix, query_prefix, updated_at "
        f"FROM {METADATA_TABLE} WHERE index_name=?",
        (INDEX_NAME,),
    ).fetchone()
    if row is None:
        return None
    return {
        "provider": row[0],
        "model": row[1],
        "dimension": int(row[2]),
        "document_prefix": row[3],
        "query_prefix": row[4],
        "updated_at": row[5],
    }


def provider_profile(provider) -> dict[str, Any]:
    return {
        "provider": provider.provider,
        "model": provider.model,
        "dimension": int(provider.dimension),
        "document_prefix": provider.document_prefix,
        "query_prefix": provider.query_prefix,
    }


def profile_mismatches(stored: dict[str, Any], configured: dict[str, Any]) -> list[str]:
    keys = ("provider", "model", "dimension", "document_prefix", "query_prefix")
    return [key for key in keys if stored.get(key) != configured.get(key)]


def write_index_metadata(conn: sqlite3.Connection, provider) -> None:
    ensure_metadata_table(conn)
    profile = provider_profile(provider)
    conn.execute(
        f"""
        INSERT INTO {METADATA_TABLE} (
            index_name, provider, model, dimension, document_prefix,
            query_prefix, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(index_name) DO UPDATE SET
            provider=excluded.provider,
            model=excluded.model,
            dimension=excluded.dimension,
            document_prefix=excluded.document_prefix,
            query_prefix=excluded.query_prefix,
            updated_at=excluded.updated_at
        """,
        (
            INDEX_NAME,
            profile["provider"],
            profile["model"],
            profile["dimension"],
            profile["document_prefix"],
            profile["query_prefix"],
            datetime.now(timezone.utc).isoformat(),
        ),
    )
