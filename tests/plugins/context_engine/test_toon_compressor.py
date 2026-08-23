# tests/plugins/context_engine/test_toon_compressor.py

import json
import unittest

from plugins.context_engine.toon_compressor import (
    ToonCompressor,
    toon_tsv,
)


# Real-shape MCP payload: array of session summaries, repeated keys.
SAMPLE_MCP = [
    {"id": 1, "name": "alice", "tokens": 1234, "active": True, "tags": ["a", "b"]},
    {"id": 2, "name": "bob", "tokens": 5678, "active": False, "tags": ["c"]},
    {"id": 3, "name": "carol", "tokens": 91011, "active": True, "tags": []},
]


class ToonTsvTests(unittest.TestCase):
    """Pure-function tests for the toon_tsv helper."""

    def test_empty_returns_none(self):
        self.assertIsNone(toon_tsv([]))

    def test_single_row(self):
        out = toon_tsv([{"a": 1, "b": "x"}])
        self.assertIn("# columns: a:int, b:str", out)
        self.assertIn("1\tx", out)

    def test_uniform_rows(self):
        out = toon_tsv(SAMPLE_MCP)
        # Header captures column order from first row
        self.assertIn("# columns: id:int, name:str, tokens:int, active:bool, tags:list", out)
        # All three data rows present
        lines = out.split("\n")
        self.assertEqual(len(lines), 4)  # header + 3 rows
        self.assertIn("1\talice\t1234\tTrue\t[\"a\", \"b\"]", lines[1])
        self.assertIn("2\tbob\t5678\tFalse\t[\"c\"]", lines[2])
        self.assertIn("3\tcarol\t91011\tTrue\t[]", lines[3])

    def test_non_uniform_returns_none(self):
        rows = [{"a": 1, "b": 2}, {"a": 1}]  # missing key in second row
        self.assertIsNone(toon_tsv(rows))

    def test_type_drift_returns_none(self):
        rows = [{"a": 1}, {"a": "string"}]  # int vs str
        self.assertIsNone(toon_tsv(rows))

    def test_extra_key_returns_none(self):
        rows = [{"a": 1, "b": 2}, {"a": 1, "b": 2, "c": 3}]
        self.assertIsNone(toon_tsv(rows))

    def test_escapes_tabs_in_strings(self):
        rows = [{"name": "tab\there"}]
        out = toon_tsv(rows)
        self.assertIn("tab\\there", out)

    def test_escapes_newlines_in_strings(self):
        rows = [{"name": "line1\nline2"}]
        out = toon_tsv(rows)
        self.assertIn("line1\\nline2", out)

    def test_handles_null_values(self):
        rows = [{"a": None, "b": "x"}]
        out = toon_tsv(rows)
        self.assertIn("\tx", out)  # null rendered as empty cell

    def test_handles_floats(self):
        rows = [{"a": 1.5, "b": "x"}]
        out = toon_tsv(rows)
        self.assertIn("1.5\tx", out)


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.eng = ToonCompressor()

    def test_name(self):
        self.assertEqual(self.eng.name, "toon_compressor")

    def test_is_available(self):
        self.assertTrue(self.eng.is_available())

    def test_empty_messages(self):
        self.assertEqual(self.eng.compress([]), [])

    def test_no_json_array_returns_unchanged(self):
        msgs = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        out = self.eng.compress(msgs)
        self.assertEqual(out, msgs)

    def test_protects_system(self):
        sys_msg = {"role": "system", "content": json.dumps(SAMPLE_MCP)}
        msgs = [sys_msg, {"role": "user", "content": "hi"}]
        out = self.eng.compress(msgs)
        self.assertEqual(out[0], sys_msg)

    def test_protects_head(self):
        head_msg = {"role": "user", "content": json.dumps(SAMPLE_MCP)}
        msgs = [
            {"role": "system", "content": "sys"},
            head_msg,
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "thanks"},
            {"role": "user", "content": json.dumps(SAMPLE_MCP)},
        ]
        out = self.eng.compress(msgs)
        self.assertEqual(out[1], head_msg)

    def test_compresses_large_array(self):
        big = json.dumps(SAMPLE_MCP + SAMPLE_MCP * 20)  # ~21x bigger
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": "y"},
            {"role": "user", "content": "z"},
            {"role": "tool", "content": big},
        ]
        out = self.eng.compress(msgs)
        new_content = out[4]["content"]
        self.assertIn("TOON summary", new_content)
        self.assertLess(len(new_content), len(big))

    def test_skips_small_array(self):
        small = json.dumps([{"a": 1}, {"a": 2}])
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": "y"},
            {"role": "user", "content": "z"},
            {"role": "tool", "content": small},
        ]
        out = self.eng.compress(msgs)
        self.assertEqual(out[4]["content"], small)

    def test_skips_invalid_json(self):
        invalid = "this is not json {]"
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": "y"},
            {"role": "user", "content": "z"},
            {"role": "tool", "content": invalid},
        ]
        out = self.eng.compress(msgs)
        self.assertEqual(out[4]["content"], invalid)

    def test_skips_non_uniform_array(self):
        nonuniform = json.dumps([{"a": 1, "b": 2}, {"a": 1}])
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": "y"},
            {"role": "user", "content": "z"},
            {"role": "tool", "content": nonuniform},
        ]
        out = self.eng.compress(msgs)
        self.assertEqual(out[4]["content"], nonuniform)

    def test_compression_count_increments(self):
        big = json.dumps(SAMPLE_MCP * 20)
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": "y"},
            {"role": "user", "content": "z"},
            {"role": "tool", "content": big},
        ]
        before = self.eng.compression_count
        self.eng.compress(msgs)
        self.assertEqual(self.eng.compression_count, before + 1)

    def test_force_flag_accepted(self):
        msgs = [{"role": "user", "content": "hi"}]
        out = self.eng.compress(msgs, force=True)
        self.assertEqual(out, msgs)

    def test_should_compress_false(self):
        self.assertFalse(self.eng.should_compress())


class PluginDiscoveryTests(unittest.TestCase):
    def test_discovered(self):
        from plugins.context_engine import discover_context_engines
        engines = discover_context_engines()
        names = [n for n, _, _ in engines]
        self.assertIn("toon_compressor", names)
        desc_by_name = {n: d for n, d, _ in engines}
        self.assertIn("TOON-style", desc_by_name["toon_compressor"])

    def test_load_engine(self):
        from plugins.context_engine import load_context_engine
        eng = load_context_engine("toon_compressor")
        self.assertIsNotNone(eng)
        self.assertIsInstance(eng, ToonCompressor)


if __name__ == "__main__":
    unittest.main()
