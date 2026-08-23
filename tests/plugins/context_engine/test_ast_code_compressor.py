# tests/plugins/context_engine/test_ast_code_compressor.py
#
# Self-tests for the ast_code_compressor ContextEngine plugin.
# Run from the Hermes checkout root:
#     PYTHONPATH=. python3 -m pytest tests/plugins/context_engine/test_ast_code_compressor.py -v
# or:
#     PYTHONPATH=. python3 -m unittest tests.plugins.context_engine.test_ast_code_compressor

import unittest
from typing import List, Dict

from plugins.context_engine.ast_code_compressor import (
    AstCodeCompressor,
    ast_skeleton,
)


VALID_PY = '''
import os
import sys
from typing import List, Dict, Optional

CONSTANT = 42
MESSAGE = "hello"

class MyClass:
    """A documented class."""

    def __init__(self, name: str) -> None:
        self.name = name

    def method(self, x: int) -> List[str]:
        return [str(x)] * 2

def top_level(a: int, b: str = "default") -> Dict[str, int]:
    return {"a": a, "b": len(b)}

async def fetch(url: str, timeout: float = 5.0) -> Optional[bytes]:
    return None
'''


class AstSkeletonTests(unittest.TestCase):
    """Tests for the pure ast_skeleton function (no engine instance)."""

    def test_returns_none_on_invalid_python(self):
        self.assertIsNone(ast_skeleton("def broken(:\n    pass"))

    def test_returns_string_on_valid_python(self):
        result = ast_skeleton(VALID_PY)
        self.assertIsNotNone(result)
        # Valid Python skeleton — re-parseable.
        import ast
        ast.parse(result)

    def test_preserves_imports(self):
        result = ast_skeleton(VALID_PY)
        self.assertIn("import os", result)
        self.assertIn("import sys", result)
        self.assertIn("from typing import List, Dict, Optional", result)

    def test_preserves_class_signature(self):
        result = ast_skeleton(VALID_PY)
        self.assertIn("class MyClass", result)
        self.assertIn("def __init__", result)
        self.assertIn("self, name: str", result)
        self.assertIn("-> None", result)

    def test_preserves_module_level_function(self):
        result = ast_skeleton(VALID_PY)
        self.assertIn("def top_level(", result)
        self.assertIn("a: int", result)
        self.assertIn("b: str = 'default'", result)
        self.assertIn("-> Dict[str, int]", result)

    def test_preserves_async_function(self):
        result = ast_skeleton(VALID_PY)
        self.assertIn("async def fetch(", result)
        self.assertIn("timeout: float = 5.0", result)

    def test_drops_function_body(self):
        result = ast_skeleton(VALID_PY)
        # The body of method() that contains [str(x)] * 2 must be dropped.
        self.assertNotIn("[str(x)] * 2", result)
        self.assertNotIn("return None", result)
        self.assertNotIn('return {"a": a', result)

    def test_replaces_body_with_pass(self):
        result = ast_skeleton(VALID_PY)
        # Method bodies should be replaced with `pass`.
        # Each non-empty method signature now ends with `: pass`.
        self.assertIn("pass", result)


class EngineDiscoveryTests(unittest.TestCase):
    """Verify the plugin is discovered by the standard discovery path."""

    def test_is_available(self):
        eng = AstCodeCompressor()
        self.assertTrue(eng.is_available())

    def test_name(self):
        eng = AstCodeCompressor()
        self.assertEqual(eng.name, "ast_code_compressor")

    def test_plugin_discovery_finds_engine(self):
        # If this import fails, the plugin isn't on the path.
        from plugins.context_engine import discover_context_engines
        engines = discover_context_engines()
        names = [name for name, _, available in engines]
        self.assertIn("ast_code_compressor", names)
        # Description should come from plugin.yaml
        desc_by_name = {name: desc for name, desc, _ in engines}
        self.assertIn("ast_code_compressor", desc_by_name)
        self.assertIn("AST-aware", desc_by_name["ast_code_compressor"])

    def test_load_engine(self):
        from plugins.context_engine import load_context_engine
        eng = load_context_engine("ast_code_compressor")
        self.assertIsNotNone(eng)
        self.assertIsInstance(eng, AstCodeCompressor)


class EngineCompressTests(unittest.TestCase):
    """End-to-end tests of the engine's compress() entry point."""

    def setUp(self):
        self.eng = AstCodeCompressor()

    def test_empty_messages_returns_empty(self):
        self.assertEqual(self.eng.compress([]), [])

    def test_no_python_blocks_returns_unchanged(self):
        msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm doing well, thanks!"},
        ]
        out = self.eng.compress(msgs)
        self.assertEqual(out, msgs)

    def test_protects_system_message(self):
        # System messages should NEVER be compressed.
        sys_msg = {"role": "system", "content": "```python\nimport os\n```"}
        msgs = [sys_msg, {"role": "user", "content": "hi"}]
        out = self.eng.compress(msgs)
        self.assertEqual(out[0], sys_msg)

    def test_protects_head_messages(self):
        # protect_first_n=3: system + first 3 non-system = first 4 messages.
        head_msg = {"role": "user", "content": "```python\n" + VALID_PY + "\n```"}
        msgs = [
            {"role": "system", "content": "sys"},
            head_msg,  # index 1 — non-system head, protected
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "thanks"},
            {"role": "user", "content": "```python\n" + VALID_PY + "\n```"},  # index 4 — compressible
        ]
        out = self.eng.compress(msgs)
        self.assertEqual(out[1], head_msg)

    def test_compresses_large_python_block(self):
        # Construct a message with a large Python block.
        big_py = "import os\n" + ("def f():\n    return 1\n" * 100) + "\n```"
        big_msg = {"role": "tool", "content": f"```python\n{big_py}"}
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "small"},
            {"role": "assistant", "content": "small"},
            {"role": "user", "content": "small"},
            big_msg,
        ]
        out = self.eng.compress(msgs)
        new_content = out[4]["content"]
        self.assertIn("AST skeleton", new_content)
        self.assertIn("import os", new_content)
        # Original body should NOT appear (only signatures).
        self.assertNotIn("return 1\n" * 50, new_content)
        self.assertLess(len(new_content), len(big_msg["content"]))

    def test_skips_small_python_blocks(self):
        # Below _MIN_BLOCK_CHARS — should pass through unchanged.
        small_py = "```python\nx = 1\ny = 2\n```"
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "x"}, {"role": "assistant", "content": "y"},
            {"role": "user", "content": "z"},
            {"role": "user", "content": small_py},
        ]
        out = self.eng.compress(msgs)
        self.assertEqual(out[4]["content"], small_py)

    def test_skips_invalid_python(self):
        # Invalid Python should pass through unchanged (not a failure).
        invalid = "```python\ndef broken(:\n    pass\n```"
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "x"}, {"role": "assistant", "content": "y"},
            {"role": "user", "content": "z"},
            {"role": "user", "content": invalid},
        ]
        out = self.eng.compress(msgs)
        self.assertEqual(out[4]["content"], invalid)

    def test_compression_count_increments(self):
        big_py = "import os\n" + ("def f():\n    return 1\n" * 100)
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "x"}, {"role": "assistant", "content": "y"},
            {"role": "user", "content": "z"},
            {"role": "tool", "content": f"```python\n{big_py}\n```"},
        ]
        before = self.eng.compression_count
        self.eng.compress(msgs)
        self.assertEqual(self.eng.compression_count, before + 1)

    def test_metrics_set_after_compression(self):
        big_py = "import os\n" + ("def f():\n    return 1\n" * 100)
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "x"}, {"role": "assistant", "content": "y"},
            {"role": "user", "content": "z"},
            {"role": "tool", "content": f"```python\n{big_py}\n```"},
        ]
        self.eng.compress(msgs)
        self.assertGreater(self.eng._last_compression_savings_pct, 50)

    def test_skips_non_text_content(self):
        # Messages with content as a list (multimodal) must NOT be touched.
        multi_msg = {"role": "user", "content": [
            {"type": "text", "text": "```python\nimport os\n" * 200 + "```"},
        ]}
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "x"}, {"role": "assistant", "content": "y"},
            {"role": "user", "content": "z"},
            multi_msg,
        ]
        out = self.eng.compress(msgs)
        self.assertEqual(out[4], multi_msg)

    def test_force_flag_accepted(self):
        # force is supported by the ABC for cooldown bypass; this engine
        # has no cooldown but the signature must accept the kwarg.
        msgs = [{"role": "user", "content": "hi"}]
        out = self.eng.compress(msgs, force=True)
        self.assertEqual(out, msgs)

    def test_focus_topic_accepted(self):
        msgs = [{"role": "user", "content": "hi"}]
        out = self.eng.compress(msgs, focus_topic="test")
        self.assertEqual(out, msgs)

    def test_should_compress_returns_false(self):
        # The engine is event-driven, not token-budget driven.
        self.assertFalse(self.eng.should_compress())
        self.assertFalse(self.eng.should_compress(150000))

    def test_update_from_response_is_noop(self):
        # Token accounting is handled by the host.
        self.eng.update_from_response({"prompt_tokens": 100, "completion_tokens": 50})
        # No assertion failure = correct behavior.


class RealHermesFileBenchmarkTests(unittest.TestCase):
    """Re-run the bench on a real Hermes file as a regression test."""

    def test_real_agent_file_compresses_to_under_15pct(self):
        import os
        # Use any large Python file from the Hermes checkout.
        candidate_paths = [
            "/Users/johnwhitman/.hermes/hermes-agent/agent/auxiliary_client.py",
            "/Users/johnwhitman/.hermes/hermes-agent/agent/agent_init.py",
        ]
        target = next((p for p in candidate_paths if os.path.exists(p)), None)
        if target is None:
            self.skipTest("No Hermes source file available")
        with open(target) as f:
            src = f.read()
        skeleton = ast_skeleton(src)
        self.assertIsNotNone(skeleton)
        ratio = len(skeleton) / len(src)
        self.assertLess(ratio, 0.15,
            f"Skeleton should be <15% of original; got {ratio:.1%} "
            f"(orig={len(src)} skel={len(skeleton)})")


if __name__ == "__main__":
    unittest.main()
