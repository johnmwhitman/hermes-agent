#!/usr/bin/env python3
"""Tests for tree-sitter multi-language AST compression (A1).

The Python path stays stdlib-only. Other languages (JS, TS, TSX, Go,
Rust) require the tree-sitter C extension + tree-sitter-languages. We
skip the multi-language tests cleanly when those deps are absent.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from plugins.context_engine.ast_code_compressor import (
    _LANGUAGE_PARSERS,
    _try_init_tree_sitter,
)


# -- Test fixtures: minimal real-looking code per language. --
JS_CODE = (
    "import x from 'y';\n"
    "function foo(a, b) {\n"
    "  const c = a + b;\n"
    "  if (c > 10) { return c; }\n"
    "  return 0;\n"
    "}\n"
    "function bar() { return 'hello'; }\n"
)
TS_CODE = (
    "interface Foo { name: string; value: number; }\n"
    "export function greet(name: string): string {\n"
    "    return 'hi ' + name;\n"
    "}\n"
)
GO_CODE = (
    "package main\n"
    "import \"fmt\"\n"
    "func add(a, b int) int { return a + b }\n"
    "func main() {\n"
    "    fmt.Println(add(1, 2))\n"
    "}\n"
)
RUST_CODE = (
    "use std::collections::HashMap;\n"
    "pub fn add(a: i32, b: i32) -> i32 {\n"
    "    let x = a + b;\n"
    "    if x > 10 { return x; }\n"
    "    return 0;\n"
    "}\n"
    "pub struct Foo { field: String }\n"
)


def _has_tree_sitter() -> bool:
    try:
        import tree_sitter  # noqa: F401
        import tree_sitter_languages  # noqa: F401
    except ImportError:
        return False
    return True


@unittest.skipUnless(_has_tree_sitter(),
                     "tree-sitter + tree-sitter-languages not installed")
class TreeSitterInitTests(unittest.TestCase):
    def test_try_init_returns_true(self):
        self.assertTrue(_try_init_tree_sitter())

    def test_languages_registered(self):
        for lang in ("javascript", "typescript", "tsx", "go", "rust"):
            self.assertIn(lang, _LANGUAGE_PARSERS,
                          f"{lang} not registered")
            parser, valid = _LANGUAGE_PARSERS[lang]
            self.assertTrue(callable(parser))
            self.assertTrue(callable(valid))


@unittest.skipUnless(_has_tree_sitter(),
                     "tree-sitter + tree-sitter-languages not installed")
class MultiLanguageCompressionTests(unittest.TestCase):
    def test_javascript_compresses(self):
        skeletonize, _ = _LANGUAGE_PARSERS["javascript"]
        out = skeletonize(JS_CODE)
        self.assertLess(len(out), len(JS_CODE),
                        "JS skeleton must be smaller than source")
        # Should preserve the function names (signatures, not bodies).
        self.assertIn("foo", out)
        self.assertIn("bar", out)

    def test_typescript_compresses(self):
        skeletonize, _ = _LANGUAGE_PARSERS["typescript"]
        out = skeletonize(TS_CODE)
        self.assertLess(len(out), len(TS_CODE))
        self.assertIn("greet", out)

    def test_go_compresses(self):
        skeletonize, _ = _LANGUAGE_PARSERS["go"]
        out = skeletonize(GO_CODE)
        self.assertLess(len(out), len(GO_CODE),
                        "Go skeleton must be smaller than source")
        self.assertIn("add", out)

    def test_rust_compresses(self):
        skeletonize, _ = _LANGUAGE_PARSERS["rust"]
        out = skeletonize(RUST_CODE)
        self.assertLess(len(out), len(RUST_CODE),
                        "Rust skeleton must be smaller than source")
        self.assertIn("add", out)
        self.assertIn("Foo", out)

    def test_empty_input_returns_empty(self):
        for lang in ("javascript", "typescript", "go", "rust"):
            skeletonize, _ = _LANGUAGE_PARSERS[lang]
            # Empty input may produce empty skeleton (no declarations).
            out = skeletonize("")
            self.assertIsInstance(out, str, f"{lang} empty input must return str")

    def test_invalid_syntax_falls_back(self):
        # Unparseable code shouldn't crash; falls back to returning source.
        for lang in ("javascript", "rust"):
            skeletonize, _ = _LANGUAGE_PARSERS[lang]
            bad_code = "this is not valid " + lang + " code ::: !!!"
            out = skeletonize(bad_code)
            self.assertIsInstance(out, str)


class MultiLanguageDiscoveryTests(unittest.TestCase):
    """Even without tree-sitter installed, engine discovery must work."""

    def test_engine_discoverable(self):
        from plugins.context_engine import load_context_engine
        engine = load_context_engine("ast_code_compressor")
        self.assertIsNotNone(engine)
        self.assertEqual(engine.name, "ast_code_compressor")

    def test_python_path_unaffected_by_multilang(self):
        """The Python (stdlib ast) path must still compress when
        tree-sitter is absent. We test the engine directly."""
        from plugins.context_engine import load_context_engine
        engine = load_context_engine("ast_code_compressor")
        big_py = (
            "import os\nimport sys\n" +
            "".join(
                f"def fn_{i}(a: int, b: str = 'x') -> int:\n"
                f"    '''Docstring for fn_{i} - this is body text.'''\n"
                f"    intermediate = a + {i}\n"
                f"    if intermediate > 100:\n"
                f"        return -1\n"
                f"    return intermediate * 2\n"
                for i in range(40)
            )
        )
        msgs = [
            {"role": "system", "content": "You are an assistant."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hi."},
            {"role": "user", "content": "Quick q."},
            {"role": "assistant", "content": "A."},
            {"role": "user", "content": "```python\n" + big_py + "\n```"},
        ]
        out = engine.select_context(msgs)
        self.assertLess(
            len(out[-1]["content"]),
            len(msgs[-1]["content"]),
            "Python compression must still work after multi-lang addition",
        )


if __name__ == "__main__":
    unittest.main()
