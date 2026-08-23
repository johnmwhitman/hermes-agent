# plugins/context_engine/ccr_store/__init__.py
#
# CCR (Compress-Cache-Retrieve) reversibility store (Hermes-native).
#
# Strategy: when a content-addressed compressor (e.g. ast_code_compressor or
# toon_compressor) replaces a block in conversation context, the original
# is content-hashed (sha256, 16-char prefix) and stored in a local SQLite
# database. The skeleton references the hash; the agent can call
# `ccr_retrieve(content_hash)` to fetch the original on demand.
#
# Compress-Cache-Retrieve (CCR) is the reversibility pattern from Headroom:
# the compression is lossless-with-original-access, never lossy. Compared to
# lossy summarization, CCR preserves information density while still
# freeing context budget.
#
# Storage: `~/.hermes/profiles/<profile>/ccr.db` (per-profile). The CCR store
# is a companion to the active context engine; it does not replace the
# engine.
#
# Configuration:
#     plugins.context_engine.ccr_store.ttl_seconds: 3600   # default 1 hour
#     plugins.context_engine.ccr_store.max_records: 10000 # cap on store size
#
# Selection: this plugin is loaded automatically when any compressing engine
# runs. It is not an engine itself; it provides tool registration for the
# `ccr_retrieve` tool.

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


DEFAULT_TTL_SECONDS = 3600
DEFAULT_MAX_RECORDS = 10000


class CcrStore:
    """Content-addressed reversibility store for compress-then-retrieve flows.

    Each record carries:
      - content_hash (sha256, hex)
      - content_type (e.g. 'python', 'json', 'text')
      - content (the original bytes)
      - byte_size
      - created_at (unix seconds)
      - last_accessed_at (unix seconds)
      - access_count
      - engine_version (str, which engine stored it)
    """

    def __init__(self, db_path: Path, ttl_seconds: int = DEFAULT_TTL_SECONDS,
                 max_records: int = DEFAULT_MAX_RECORDS):
        self.db_path = Path(db_path)
        self.ttl_seconds = ttl_seconds
        self.max_records = max_records
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=10, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_schema(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS records (
                    content_hash TEXT PRIMARY KEY,
                    content_type TEXT NOT NULL,
                    content BLOB NOT NULL,
                    byte_size INTEGER NOT NULL,
                    created_at INTEGER NOT NULL,
                    last_accessed_at INTEGER NOT NULL,
                    access_count INTEGER NOT NULL DEFAULT 0,
                    engine_version TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS records_created_at_idx
                ON records(created_at)
            """)

    # -- Public API --------------------------------------------------------

    def put(self, content: bytes, content_type: str, engine_version: str) -> str:
        """Store content under its sha256 hash. Returns the 16-char hash prefix.

        Idempotent: re-storing the same content updates last_accessed_at
        but not created_at. If the store is at capacity, evicts the oldest
        records (LRU).
        """
        content_hash = hashlib.sha256(content).hexdigest()
        short = content_hash[:16]
        now = int(time.time())
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO records (content_hash, content_type, content, byte_size,
                                     created_at, last_accessed_at, access_count,
                                     engine_version)
                VALUES (?, ?, ?, ?, ?, ?, 0, ?)
                ON CONFLICT(content_hash) DO UPDATE SET
                    last_accessed_at = excluded.last_accessed_at
                """,
                (content_hash, content_type, content, len(content),
                 now, now, engine_version),
            )
        self._enforce_capacity()
        return short

    def get(self, short_hash: str) -> Optional[Dict[str, Any]]:
        """Retrieve a record by its 16-char hash prefix. Returns None if missing
        or expired. Updates access_count and last_accessed_at.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT content_hash, content_type, content, byte_size, created_at, "
                "last_accessed_at, access_count, engine_version "
                "FROM records WHERE content_hash LIKE ? LIMIT 1",
                (short_hash + "%",),
            ).fetchone()
        if row is None:
            return None
        content_hash, content_type, content, byte_size, created_at, \
            last_accessed_at, access_count, engine_version = row
        now = int(time.time())
        if now - created_at > self.ttl_seconds:
            self._evict(content_hash)
            return None
        # Bump access counters.
        with self._connect() as conn:
            conn.execute(
                "UPDATE records SET last_accessed_at = ?, access_count = access_count + 1 "
                "WHERE content_hash = ?",
                (now, content_hash),
            )
        return {
            "content_hash": content_hash,
            "short_hash": content_hash[:16],
            "content_type": content_type,
            "content": content,
            "byte_size": byte_size,
            "created_at": created_at,
            "last_accessed_at": now,
            "access_count": access_count + 1,
            "engine_version": engine_version,
        }

    def stats(self) -> Dict[str, Any]:
        """Return store statistics for telemetry."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*), SUM(byte_size) FROM records"
            ).fetchone()
            count, total_bytes = row
        return {
            "record_count": count or 0,
            "total_bytes": total_bytes or 0,
            "ttl_seconds": self.ttl_seconds,
            "max_records": self.max_records,
        }

    # -- Internal helpers --------------------------------------------------

    def _evict(self, content_hash: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM records WHERE content_hash = ?",
                         (content_hash,))

    def _enforce_capacity(self) -> None:
        """LRU-evict oldest records until count <= max_records."""
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM records").fetchone()
            count = row[0] if row else 0
        if count <= self.max_records:
            return
        # Evict the (count - max_records) oldest by last_accessed_at.
        to_evict = count - self.max_records
        with self._connect() as conn:
            victims = conn.execute(
                "SELECT content_hash FROM records ORDER BY last_accessed_at ASC LIMIT ?",
                (to_evict,),
            ).fetchall()
            for (h,) in victims:
                conn.execute("DELETE FROM records WHERE content_hash = ?", (h,))


# Module-level singleton — instantiated lazily so each Hermes profile gets
# its own DB path under ~/.hermes/profiles/<profile>/ccr.db.
_STORE: Optional[CcrStore] = None
_STORE_PATH: Optional[Path] = None


def get_store(profile_dir: Optional[Path] = None) -> CcrStore:
    """Return the CCR store for the given profile directory (or the default).

    Default: `~/.hermes/profiles/conductor/ccr.db`. Override with the
    `profile_dir` argument.
    """
    global _STORE, _STORE_PATH
    if profile_dir is None:
        profile_dir = Path.home() / ".hermes" / "profiles" / "conductor"
    db_path = Path(profile_dir) / "ccr.db"
    if _STORE is None or _STORE_PATH != db_path:
        _STORE = CcrStore(db_path)
        _STORE_PATH = db_path
    return _STORE


def ccr_retrieve(short_hash: str, profile_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Retrieve a stored content record by 16-char sha256 prefix.

    Returns a dict with content (bytes) and metadata, or an error dict.
    This is the tool surface exposed to the agent via plugin tooling.
    """
    store = get_store(profile_dir)
    rec = store.get(short_hash)
    if rec is None:
        return {"error": "not_found_or_expired", "short_hash": short_hash}
    return {
        "content_hash": rec["content_hash"],
        "content_type": rec["content_type"],
        "content": rec["content"].decode("utf-8", errors="replace"),
        "byte_size": rec["byte_size"],
        "created_at": rec["created_at"],
        "access_count": rec["access_count"],
        "engine_version": rec["engine_version"],
    }
