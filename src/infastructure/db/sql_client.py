"""SQLAlchemy session utilities for Supabase PostgreSQL."""

from __future__ import annotations

from typing import Optional

try:
    from sqlalchemy import Column, DateTime, Integer, MetaData, Table, Text, create_engine, text
    from sqlalchemy.orm import sessionmaker
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    Column = DateTime = Integer = MetaData = Table = Text = create_engine = text = sessionmaker = None

from infastructure.config import SUPABASE_DB_URL


metadata = MetaData() if MetaData is not None else None

st_turns_table = (
    Table(
        "st_turns",
        metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("user_id", Text, nullable=False),
        Column("session_id", Text, nullable=False),
        Column("role", Text, nullable=False),
        Column("content", Text, nullable=False),
        Column("created_at", DateTime(timezone=True), nullable=False, server_default=text("NOW()")),
        Column("ttl_at", DateTime(timezone=True), nullable=True),
    )
    if Table is not None
    else None
)

_engine: Optional[object] = None
_session_factory: Optional[object] = None


def get_sql_engine():
    """Return a singleton SQLAlchemy engine for Supabase PostgreSQL."""
    global _engine
    if _engine is not None:
        return _engine

    if create_engine is None:
        raise RuntimeError("sqlalchemy is required for Supabase SQL support.")

    if not SUPABASE_DB_URL:
        raise ValueError("SUPABASE_DB_URL is not configured.")

    _engine = create_engine(
        SUPABASE_DB_URL,
        pool_pre_ping=True,
        pool_recycle=3600,
        future=True,
    )
    return _engine


def get_session():
    """Return a new SQLAlchemy session."""
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(bind=get_sql_engine(), autoflush=False, autocommit=False, future=True)
    return _session_factory()


def create_tables() -> None:
    """Create the short-term memory table if it does not exist."""
    if metadata is None:
        raise RuntimeError("sqlalchemy is required for Supabase SQL support.")
    metadata.create_all(bind=get_sql_engine())
