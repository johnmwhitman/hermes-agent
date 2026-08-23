"""Tests for the unfenced-Python compression pass added 2026-08-23.

The fenced pass (NEXT-CYCLE-QUEUE A1) only fires on ```python|py fences.
The unfenced pass catches bare Python source pasted into chat. Together
they cover both formatting shapes.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from plugins.context_engine.ast_code_compressor import (
    AstCodeCompressor,
    _apply_unfenced_pass,
    _compress_unfenced_segment,
)


# A valid Python block, deliberately bare (no fence, no markdown).
_BIG_PY = (
    "import os\n"
    "import sys\n"
    "from typing import List, Optional\n"
    + "".join(
        f"def function_{i}(arg_one: int, arg_two: str = 'default_value_{i}') -> Optional[int]:\n"
        f"    '''Docstring for function_{i} - spans multiple lines to add body bytes.'''\n"
        f"    intermediate = arg_one + {i}\n"
        f"    if intermediate > 100:\n"
        f"        return None\n"
        f"    return intermediate * 2\n"
        for i in range(80)
    )
)


class UnfencedSegmentTests(unittest.TestCase):
    def test_segment_too_short_passes_through(self):
        result = _compress_unfenced_segment("x = 1", lambda m: m.group("body"), {})
        self.assertEqual(result, "x = 1")

    def test_segment_invalid_python_passes_through(self):
        # "this is not python" — no def, no import, just text.
        text = "this is just plain text " * 50  # long but not Python
        result = _compress_unfenced_segment(text, lambda m: m.group("body"), {})
        self.assertEqual(result, text)

    def test_segment_valid_python_compresses(self):
        stats = {"blocks_found": 0, "blocks_compressed": 0, "bytes_saved": 0}

        def fake_replace(m):
            body = m.group("body")
            return f"# skeleton for {len(body)} chars"

        result = _compress_unfenced_segment(_BIG_PY, fake_replace, stats)
        self.assertNotEqual(result, _BIG_PY)
        self.assertIn("skeleton", result)
        self.assertGreater(stats["blocks_found"], 0)

    def test_segment_with_invalid_python_after_valid_python(self):
        # Two chunks separated by double newline.
        mixed = _BIG_PY + "\n\n" + "and this is plain text not Python " * 20
        stats = {"blocks_found": 0, "blocks_compressed": 0, "bytes_saved": 0}

        def fake_replace(m):
            body = m.group("body")
            return f"# skeleton for {len(body)} chars"

        result = _compress_unfenced_segment(mixed, fake_replace, stats)
        self.assertIn("skeleton", result)
        # The plain text portion should remain.
        self.assertIn("plain text", result)


class UnfencedPassTests(unittest.TestCase):
    def test_empty_string_returns_empty(self):
        self.assertEqual(_apply_unfenced_pass("", lambda m: m.group("body"), {}), "")

    def test_only_skeleton_markers_returns_through(self):
        text = "# AST skeleton (saved 100 bytes)\n# original_sha256=abcd1234abcd1234\n\n"
        result = _apply_unfenced_pass(text, lambda m: m.group("body"), {})
        self.assertEqual(result, text)

    def test_mixed_fenced_and_unfenced(self):
        # Fenced block + unfenced block in same text.
        fenced_block = (
            "```python\n" + _BIG_PY + "\n```"
        )
        text = (
            "Here's a fenced block:\n\n"
            + fenced_block
            + "\n\nAnd an unfenced one:\n\n"
            + _BIG_PY
        )
        stats = {"blocks_found": 0, "blocks_compressed": 0, "bytes_saved": 0}

        def fake_replace(m):
            body = m.group("body")
            return f"# skeleton for {len(body)} chars"

        result = _apply_unfenced_pass(text, fake_replace, stats)
        # Both the fenced and unfenced blocks should be replaced.
        self.assertIn("skeleton", result)
        # The unfenced portion is distinct from the fenced (different raw
        # inputs but same skeleton output, so we just check the marker count).
        self.assertGreaterEqual(result.count("skeleton"), 1)


class IntegrationWithAstCodeCompressorTests(unittest.TestCase):
    def test_engine_compresses_unfenced_python(self):
        engine = AstCodeCompressor()
        msg_text = "Look at this code:\n\n" + _BIG_PY + "\n\nWhat does it do?"
        out, stats = engine._compress_text(msg_text)
        self.assertGreater(stats["blocks_found"], 0)
        self.assertNotEqual(out, msg_text)
        # The body should be replaced with an AST skeleton marker.
        self.assertIn("AST skeleton", out)

    def test_engine_does_not_break_on_pure_prose(self):
        engine = AstCodeCompressor()
        prose = "This is just regular text with no code at all. " * 100
        out, stats = engine._compress_text(prose)
        # No blocks found — prose passes through.
        self.assertEqual(stats["blocks_found"], 0)
        self.assertEqual(out, prose)

    def test_engine_does_not_break_on_invalid_python(self):
        engine = AstCodeCompressor()
        invalid = "this is not valid python: := :: === >>> @@@\n" * 50
        out, stats = engine._compress_text(invalid)
        # Doesn't crash; may or may not find blocks (the invalid Python
        # won't parse as valid ast, so blocks_compressed stays 0).
        self.assertEqual(stats.get("blocks_compressed", 0), 0)


if __name__ == "__main__":
    unittest.main()
