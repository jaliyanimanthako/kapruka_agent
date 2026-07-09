"""Supabase client helpers for the short-term memory store."""

from __future__ import annotations

from typing import Any, Optional

try:
    from supabase import Client, create_client
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    Client = Any
    create_client = None

from infastructure.config import SUPABASE_ANON_KEY, SUPABASE_DB_URL, SUPABASE_URL
from infastructure.db.sql_client import create_tables, get_session, get_sql_engine


_supabase_client: Optional[Client] = None


def get_supabase_client() -> Client:
    """Return a singleton Supabase REST client."""
    global _supabase_client
    if _supabase_client is not None:
        return _supabase_client

    if create_client is None:
        raise RuntimeError("supabase package is required for Supabase REST support.")

    if not SUPABASE_URL or not SUPABASE_ANON_KEY:
        raise ValueError("SUPABASE_URL and SUPABASE_ANON_KEY must be configured.")

    _supabase_client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    return _supabase_client


def get_supabase_engine():
    """Expose the SQLAlchemy engine used for Supabase PostgreSQL."""
    return get_sql_engine()


def get_supabase_session():
    """Expose the SQLAlchemy session factory used for Supabase PostgreSQL."""
    return get_session()


def supabase_available() -> bool:
    """Check whether both REST and SQL connectivity can be configured."""
    return bool(SUPABASE_URL and SUPABASE_ANON_KEY and SUPABASE_DB_URL)


def init_supabase_schema() -> None:
    """Create the minimal short-term-memory schema."""
    create_tables()
