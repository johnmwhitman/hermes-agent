# tests/plugins/context_engine/test_ccr_store.py

import os
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path

from plugins.context_engine.ccr_store import (
    CcrStore,
    ccr_retrieve,
    get_store,
)


class CcrStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = Path(self.tmpdir) / "test_ccr.db"
        self.store = CcrStore(self.db_path)

    def test_store_and_retrieve(self):
        h = self.store.put(b"hello world", "text", "v1")
        rec = self.store.get(h)
        self.assertIsNotNone(rec)
        self.assertEqual(rec["content"], b"hello world")
        self.assertEqual(rec["content_type"], "text")
        self.assertEqual(rec["engine_version"], "v1")

    def test_short_hash_is_16_chars(self):
        h = self.store.put(b"x", "text", "v1")
        self.assertEqual(len(h), 16)

    def test_full_hash_in_record(self):
        h = self.store.put(b"y", "text", "v1")
        rec = self.store.get(h)
        self.assertEqual(len(rec["content_hash"]), 64)
        self.assertTrue(rec["content_hash"].startswith(h))

    def test_idempotent_put(self):
        h1 = self.store.put(b"same", "text", "v1")
        h2 = self.store.put(b"same", "text", "v1")
        self.assertEqual(h1, h2)
        # Re-put updates last_accessed_at but not access_count (which only
        # bumps on get()). Verify by reading access_count without get().
        with sqlite3.connect(str(self.db_path)) as conn:
            row = conn.execute(
                "SELECT access_count FROM records WHERE content_hash LIKE ?",
                (h1 + "%",),
            ).fetchone()
        self.assertEqual(row[0], 0)

    def test_access_count_increments(self):
        h = self.store.put(b"x", "text", "v1")
        self.store.get(h)
        rec = self.store.get(h)
        self.assertEqual(rec["access_count"], 2)

    def test_retrieve_nonexistent_returns_none(self):
        rec = self.store.get("deadbeef")
        self.assertIsNone(rec)

    def test_ttl_expiry(self):
        # Use a 1-second TTL and sleep well past it
        store = CcrStore(self.db_path, ttl_seconds=1)
        h = store.put(b"expires", "text", "v1")
        time.sleep(2.2)
        rec = store.get(h)
        self.assertIsNone(rec)

    def test_lru_eviction_at_capacity(self):
        store = CcrStore(self.db_path, max_records=2)
        store.put(b"a", "text", "v1")
        time.sleep(0.01)
        store.put(b"b", "text", "v1")
        time.sleep(0.01)
        # Touch 'a' so it's more recently accessed
        store.get(store.put(b"a", "text", "v1"))
        # Insert a third — should evict 'b' (older last_accessed_at)
        store.put(b"c", "text", "v1")
        stats = store.stats()
        self.assertEqual(stats["record_count"], 2)

    def test_stats(self):
        self.store.put(b"a", "text", "v1")
        self.store.put(b"bb", "text", "v1")
        stats = self.store.stats()
        self.assertEqual(stats["record_count"], 2)
        self.assertEqual(stats["total_bytes"], 3)
        self.assertEqual(stats["ttl_seconds"], 3600)

    def test_schema_persists_across_reopen(self):
        self.store.put(b"x", "text", "v1")
        store2 = CcrStore(self.db_path)
        rec = store2.get(self.store.put(b"x", "text", "v1"))
        self.assertIsNotNone(rec)

    def test_byte_size_recorded(self):
        h = self.store.put(b"abcdefghij", "text", "v1")
        rec = self.store.get(h)
        self.assertEqual(rec["byte_size"], 10)

    def test_wal_journal_mode(self):
        with sqlite3.connect(str(self.db_path)) as conn:
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        self.assertEqual(mode.lower(), "wal")


class ModuleLevelStoreTests(unittest.TestCase):
    def test_get_store_singleton(self):
        s1 = get_store(Path(tempfile.mkdtemp()))
        s2 = get_store(Path(tempfile.mkdtemp()))  # different path
        # Different paths -> different instances
        self.assertIsNot(s1, s2)

    def test_ccr_retrieve_tool(self):
        tmpdir = Path(tempfile.mkdtemp())
        # Reset the module-level singleton
        import plugins.context_engine.ccr_store as ccr_mod
        ccr_mod._STORE = None
        ccr_mod._STORE_PATH = None
        store = get_store(tmpdir)
        h = store.put(b"hello via tool", "text", "test-v1")
        result = ccr_retrieve(h, profile_dir=tmpdir)
        self.assertNotIn("error", result)
        self.assertEqual(result["content"], "hello via tool")


if __name__ == "__main__":
    unittest.main()
