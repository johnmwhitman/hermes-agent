"""Tests for the ccr_retrieve tool registration on the ccr_store plugin."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from plugins.context_engine.ccr_store import (
    CCR_RETRIEVE_SCHEMA,
    CcrStore,
    register,
    ccr_retrieve,
)


class _StubCtx:
    def __init__(self):
        self.tools = []

    def register_tool(self, *, name, toolset, schema, handler, description,
                      emoji, override=False):
        self.tools.append({
            "name": name, "toolset": toolset, "schema": schema,
            "handler": handler, "description": description,
            "emoji": emoji, "override": override,
        })


class RegisterTests(unittest.TestCase):
    def test_register_registers_ccr_retrieve_in_ccr_toolset(self):
        ctx = _StubCtx()
        register(ctx)
        self.assertEqual(len(ctx.tools), 1)
        t = ctx.tools[0]
        self.assertEqual(t["name"], "ccr_retrieve")
        self.assertEqual(t["toolset"], "ccr")
        self.assertTrue(t["override"], "must override any stale handler")

    def test_schema_has_required_short_hash(self):
        ctx = _StubCtx()
        register(ctx)
        schema = ctx.tools[0]["schema"]
        params = schema["parameters"]
        self.assertEqual(params["type"], "object")
        self.assertIn("short_hash", params["properties"])
        self.assertEqual(params["properties"]["short_hash"]["type"], "string")
        self.assertIn("short_hash", params["required"])

    def test_schema_also_documents_profile_dir(self):
        ctx = _StubCtx()
        register(ctx)
        props = ctx.tools[0]["schema"]["parameters"]["properties"]
        self.assertIn("profile_dir", props)
        self.assertEqual(props["profile_dir"]["type"], "string")

    def test_schema_description_mentions_the_marker_pattern(self):
        ctx = _StubCtx()
        register(ctx)
        desc = ctx.tools[0]["description"]
        self.assertIn("original_sha256", desc)
        self.assertIn("ccr_retrieve", desc)

    def test_schema_is_exportable(self):
        ctx = _StubCtx()
        register(ctx)
        self.assertEqual(
            ctx.tools[0]["schema"],
            CCR_RETRIEVE_SCHEMA["function"],
        )


class HandlerTests(unittest.TestCase):
    """Each test gets its own profile_dir so the singleton can't leak."""

    def setUp(self):
        self.profile_dir = Path(tempfile.mkdtemp())

    def _handler(self, ctx):
        return ctx.tools[0]["handler"]

    def test_handler_retrieves_stored_content(self):
        # Build a store at our temp profile dir; pass profile_dir through
        # the handler so it reads from the same DB.
        store = CcrStore(self.profile_dir / "ccr.db")
        h = store.put(b"hello world", "text", "v1")
        ctx = _StubCtx()
        register(ctx)
        result = self._handler(ctx)(short_hash=h, profile_dir=str(self.profile_dir))
        self.assertNotIn("error", result)
        self.assertEqual(result["content"], "hello world")
        self.assertEqual(result["content_type"], "text")

    def test_handler_returns_structured_error_for_unknown_hash(self):
        ctx = _StubCtx()
        register(ctx)
        result = self._handler(ctx)(
            short_hash="ffffffffffffffff",
            profile_dir=str(self.profile_dir),
        )
        self.assertEqual(result["error"], "not_found_or_expired")
        self.assertEqual(result["short_hash"], "ffffffffffffffff")

    def test_handler_truncates_oversized_content(self):
        store = CcrStore(self.profile_dir / "ccr.db")
        big = b"x" * 25000  # > 20000 truncation threshold
        h = store.put(big, "text", "v1")
        ctx = _StubCtx()
        register(ctx)
        result = self._handler(ctx)(short_hash=h, profile_dir=str(self.profile_dir))
        self.assertTrue(result.get("truncated", False))
        self.assertIn("TRUNCATED at 20000 chars", result["content"])
        self.assertEqual(result["byte_size"], 25000)

    def test_handler_falls_back_to_hermes_home_env(self):
        """When profile_dir is omitted, HERMES_HOME drives the resolution."""
        store = CcrStore(self.profile_dir / "ccr.db")
        h = store.put(b"env-fallback", "text", "v1")
        ctx = _StubCtx()
        register(ctx)
        # Note: handler reads HERMES_HOME itself; we pre-set it via monkeypatch.
        import os
        old = os.environ.get("HERMES_HOME")
        os.environ["HERMES_HOME"] = str(self.profile_dir)
        try:
            result = self._handler(ctx)(short_hash=h)
            self.assertNotIn("error", result)
            self.assertEqual(result["content"], "env-fallback")
        finally:
            if old is None:
                os.environ.pop("HERMES_HOME", None)
            else:
                os.environ["HERMES_HOME"] = old

    def test_module_level_ccr_retrieve_still_works(self):
        store = CcrStore(self.profile_dir / "ccr.db")
        h = store.put(b"direct-call", "text", "v1")
        result = ccr_retrieve(h, profile_dir=self.profile_dir)
        self.assertEqual(result["content"], "direct-call")


if __name__ == "__main__":
    unittest.main()
