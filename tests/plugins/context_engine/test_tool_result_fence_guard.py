"""Guard tests: tool roles and bare fences must not be treated as Python."""
import unittest

from plugins.context_engine.ast_code_compressor import (
    AstCodeCompressor,
    _FENCED_PY_RE,
)


class ToolResultAndFenceGuardTests(unittest.TestCase):
    def test_tool_role_is_left_alone(self):
        engine = AstCodeCompressor()
        blob = "import os\n" + "".join(
            "def handler_%d(x):\n    return x * %d\n" % (i, i) for i in range(80)
        )
        original = "```python\n" + blob + "\n```"
        msg = {"role": "tool", "content": original}
        out, stats = engine._process_message(msg)
        self.assertEqual(out["content"], original)
        self.assertEqual(stats.get("blocks_compressed", 0), 0)

    def test_bare_fence_is_not_treated_as_python(self):
        text = "HEALTH 200\n```\nstatus ok\n```\n"
        self.assertIsNone(_FENCED_PY_RE.search(text))
        engine = AstCodeCompressor()
        msg = {"role": "assistant", "content": text}
        out, stats = engine._process_message(msg)
        self.assertEqual(out["content"], text)
        self.assertEqual(stats.get("blocks_compressed", 0), 0)

    def test_explicit_python_fence_still_compresses(self):
        engine = AstCodeCompressor()
        blob = "import os\n" + "".join(
            "def handler_%d(x):\n    return x * %d\n" % (i, i) for i in range(80)
        )
        content = "```python\n" + blob + "\n```"
        msgs = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
            {"role": "assistant", "content": "d"},
            {"role": "user", "content": content},
        ]
        out = engine.compress(msgs)
        self.assertLess(len(out[-1]["content"]), len(content))
        self.assertIn("AST skeleton", out[-1]["content"])


if __name__ == "__main__":
    unittest.main()
