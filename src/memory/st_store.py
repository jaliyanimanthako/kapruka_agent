"""Short-term memory store backed by Supabase PostgreSQL, with local fallback."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from typing import Deque, Dict, List, Tuple

from infastructure.config import ST_MAX_TURNS, ST_TTL_SECONDS, SUPABASE_DB_URL
from infastructure.db.sql_client import create_tables, get_session, st_turns_table
from memory.schemas import ConversationTurn


class ShortTermMemoryStore:
    """Session-scoped conversation buffer for live context."""

    def __init__(
        self,
        max_turns: int = ST_MAX_TURNS,
        ttl_seconds: int = ST_TTL_SECONDS,
        use_database: bool | None = None,
    ) -> None:
        self.max_turns = max_turns
        self.ttl_seconds = ttl_seconds
        self._use_database = bool(SUPABASE_DB_URL) if use_database is None else use_database
        self._db_ready = False
        self._local_store: Dict[Tuple[str, str], Deque[ConversationTurn]] = defaultdict(deque)

    def append(self, turn: ConversationTurn) -> None:
        """Append one turn into short-term memory."""
        if self._use_database:
            self._ensure_database_ready()
        if self._use_database:
            self._append_db(turn)
            return
        self._append_local(turn)

    def recent(self, user_id: str, session_id: str, k: int = 6) -> List[ConversationTurn]:
        """Return recent turns for the given user session."""
        if self._use_database:
            self._ensure_database_ready()
        if self._use_database:
            return self._recent_db(user_id, session_id, k)
        return self._recent_local(user_id, session_id, k)

    def clear(self, user_id: str, session_id: str) -> None:
        """Remove all short-term turns for a session."""
        if self._use_database:
            self._ensure_database_ready()
        if self._use_database:
            session = get_session()
            try:
                session.execute(
                    delete(st_turns_table).where(
                        st_turns_table.c.user_id == user_id,
                        st_turns_table.c.session_id == session_id,
                    )
                )
                session.commit()
            finally:
                session.close()
            return

        self._local_store.pop((user_id, session_id), None)

    def _ensure_database_ready(self) -> None:
        if not self._use_database or self._db_ready:
            return
        try:
            create_tables()
            self._db_ready = True
        except Exception:
            self._use_database = False

    def _append_local(self, turn: ConversationTurn) -> None:
        key = (turn.user_id, turn.session_id)
        bucket = self._local_store[key]
        bucket.append(turn)
        self._trim_local(bucket)

    def _recent_local(self, user_id: str, session_id: str, k: int) -> List[ConversationTurn]:
        key = (user_id, session_id)
        bucket = self._local_store.get(key, deque())
        self._trim_local(bucket)
        return list(bucket)[-k:]

    def _trim_local(self, bucket: Deque[ConversationTurn]) -> None:
        cutoff = time.time() - self.ttl_seconds
        while bucket and bucket[0].ts < cutoff:
            bucket.popleft()
        while len(bucket) > self.max_turns:
            bucket.popleft()

    def _append_db(self, turn: ConversationTurn) -> None:
        from sqlalchemy import delete, insert, select

        session = get_session()
        try:
            ttl_at = datetime.now(timezone.utc) + timedelta(seconds=self.ttl_seconds)
            session.execute(
                insert(st_turns_table).values(
                    user_id=turn.user_id,
                    session_id=turn.session_id,
                    role=turn.role,
                    content=turn.content,
                    ttl_at=ttl_at,
                )
            )
            session.commit()

            rows = session.execute(
                select(st_turns_table.c.id)
                .where(
                    st_turns_table.c.user_id == turn.user_id,
                    st_turns_table.c.session_id == turn.session_id,
                )
                .order_by(st_turns_table.c.created_at.desc())
            ).fetchall()
            stale_ids = [row.id for row in rows[self.max_turns :]]
            if stale_ids:
                session.execute(delete(st_turns_table).where(st_turns_table.c.id.in_(stale_ids)))
                session.commit()
        finally:
            session.close()

    def _recent_db(self, user_id: str, session_id: str, k: int) -> List[ConversationTurn]:
        from sqlalchemy import select

        session = get_session()
        try:
            now = datetime.now(timezone.utc)
            rows = session.execute(
                select(
                    st_turns_table.c.role,
                    st_turns_table.c.content,
                    st_turns_table.c.created_at,
                )
                .where(
                    st_turns_table.c.user_id == user_id,
                    st_turns_table.c.session_id == session_id,
                    (st_turns_table.c.ttl_at.is_(None)) | (st_turns_table.c.ttl_at > now),
                )
                .order_by(st_turns_table.c.created_at.desc())
                .limit(k)
            ).fetchall()
            return [
                ConversationTurn(
                    user_id=user_id,
                    session_id=session_id,
                    role=row.role,
                    content=row.content,
                    ts=row.created_at.timestamp(),
                )
                for row in reversed(rows)
            ]
        finally:
            session.close()
