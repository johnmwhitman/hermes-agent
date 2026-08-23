# tests/plugins/context_engine/test_select_context_hook.py
#
# B3 follow-up (2026-08-23, second wave): the AST and TOON engines were
# shipped with should_compress() returning False, which meant the host
# never called compress(). Without a per-turn hook, the engines were
# dormant in production. This test pins the new select_context() method
# which the host calls every turn independently of should_compress().

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from plugins.context_engine import load_context_engine


# A large Python block (must exceed _MIN_BLOCK_CHARS - 200 in the engine)
_BIG_PY = (
    'import os\nimport sys\nfrom typing import List, Dict\n'
    + ('def f(x: int, y: int = 5) -> List[Dict[str, int]]:\n    """Docstring."""\n'
       '    return [{"x": x, "y": y}]\n' * 20)
    + '\n'
)

_BIG_JSON = '{"items": [' + ','.join(
    f'{{"id": {i}, "name": "thing_{i}", "value": {i*2}}}'
    for i in range(200)
) + ']}'


def _msgs_with_block(block: str, fence_lang: str = 'python'):
    """Build a message list where the heavy block is at an UNPROTECTED index.

    protect_first_n=3 (the ABC default for our engines) means the host
    protects system + first 3 non-system messages. The block must land at
    index >= 4 to be eligible for compression.
    """
    return [
        {'role': 'system', 'content': 'You are an assistant.'},
        {'role': 'user', 'content': 'Hi'},
        {'role': 'assistant', 'content': 'Hello!'},
        {'role': 'user', 'content': 'Quick question first.'},
        {'role': 'assistant', 'content': 'Sure, what?'},
        {'role': 'user', 'content': f'Read this:\n\n```{fence_lang}\n{block}\n```'},
        {'role': 'assistant', 'content': 'Analyzing.'},
        {'role': 'user', 'content': 'Also: ' + 'X' * 5000},
    ]


class AstCodeCompressorSelectContextTests(unittest.TestCase):
    def test_select_context_fires_per_turn(self):
        engine = load_context_engine('ast_code_compressor')
        self.assertIsNotNone(engine, "ast_code_compressor not loadable")
        msgs = _msgs_with_block(_BIG_PY)

        # The base ABC select_context is no-op; the engine override must
        # actually call _compress_messages and return a compressed list.
        self.assertTrue(hasattr(engine, 'select_context'))
        before_len = sum(len(m['content']) for m in msgs)
        out = engine.select_context(msgs)
        after_len = sum(len(m['content']) for m in out)
        self.assertLess(after_len, before_len,
                        f"select_context did not compress ({before_len} -> {after_len})")

    def test_select_context_does_not_mutate_input(self):
        engine = load_context_engine('ast_code_compressor')
        msgs = _msgs_with_block(_BIG_PY)
        before_snapshot = [dict(m) for m in msgs]  # shallow copy
        before_deep = [m['content'] for m in msgs]
        engine.select_context(msgs)
        for i, m in enumerate(msgs):
            self.assertEqual(m['content'], before_deep[i],
                             f"msg[{i}] content mutated in place")

    def test_select_context_preserves_protected_head(self):
        engine = load_context_engine('ast_code_compressor')
        msgs = _msgs_with_block(_BIG_PY)
        out = engine.select_context(msgs)
        # System + first 3 non-system = 4 messages should be preserved verbatim.
        # In our fixture that's indices 0..3 (system, user, assistant, user).
        for i in range(4):
            self.assertEqual(out[i]['content'], msgs[i]['content'],
                             f"head msg[{i}] unexpectedly modified")

    def test_select_context_with_ccr_store_round_trip(self):
        engine = load_context_engine('ast_code_compressor')
        db_path = Path(tempfile.mkdtemp()) / 'ccr.sqlite'
        from plugins.context_engine.ccr_store import CcrStore
        store = CcrStore(db_path)
        engine.set_ccr_store(store)

        msgs = _msgs_with_block(_BIG_PY)
        out = engine.select_context(msgs)
        # At least one msg should now have a skeleton with original_sha256
        skeletons = [m['content'] for m in out if 'original_sha256' in m.get('content', '')]
        self.assertGreater(len(skeletons), 0, "no skeleton produced")
        # And the ccr store should have the original
        short_hash = [line.split('original_sha256=')[1].split('\n')[0]
                      for line in skeletons[0].split('\n')
                      if 'original_sha256=' in line][0]
        rec = store.get(short_hash)
        self.assertIsNotNone(rec, "CCR store missing the original")
        self.assertIn(b'def f(x: int', rec['content'])


class ToonCompressorSelectContextTests(unittest.TestCase):
    def test_select_context_fires_per_turn(self):
        engine = load_context_engine('toon_compressor')
        self.assertIsNotNone(engine, "toon_compressor not loadable")
        msgs = _msgs_with_block(_BIG_JSON, fence_lang='json')

        before_len = sum(len(m['content']) for m in msgs)
        out = engine.select_context(msgs)
        after_len = sum(len(m['content']) for m in out)
        self.assertLess(after_len, before_len,
                        f"select_context did not compress ({before_len} -> {after_len})")

    def test_select_context_preserves_protected_head(self):
        engine = load_context_engine('toon_compressor')
        msgs = _msgs_with_block(_BIG_JSON, fence_lang='json')
        out = engine.select_context(msgs)
        for i in range(4):
            self.assertEqual(out[i]['content'], msgs[i]['content'],
                             f"head msg[{i}] unexpectedly modified")

    def test_select_context_non_array_json_is_passed_through(self):
        engine = load_context_engine('toon_compressor')
        # Single dict (not an array) - TOON only handles arrays of uniform dicts
        msgs = [
            {'role': 'system', 'content': 'sys'},
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'content': 'a'},
            {'role': 'user', 'content': '```json\n{"single": "object", "not": "array"}\n```'},
        ]
        before = msgs[3]['content']
        out = engine.select_context(msgs)
        # The single-object JSON should pass through (TOON doesn't compress non-arrays)
        self.assertEqual(out[3]['content'], before)


class CompressorStillWorksViaCompressMethod(unittest.TestCase):
    """Make sure the refactor (extract _compress_messages) didn't break
    the explicit compress() path. The host still calls this for token-
    threshold-driven compaction, even though our engines always return
    False from should_compress()."""

    def test_ast_compress_method_still_works(self):
        engine = load_context_engine('ast_code_compressor')
        msgs = _msgs_with_block(_BIG_PY)
        out = engine.compress(msgs)
        before_len = sum(len(m['content']) for m in msgs)
        after_len = sum(len(m['content']) for m in out)
        self.assertLess(after_len, before_len)

    def test_toon_compress_method_still_works(self):
        engine = load_context_engine('toon_compressor')
        msgs = _msgs_with_block(_BIG_JSON, fence_lang='json')
        out = engine.compress(msgs)
        before_len = sum(len(m['content']) for m in msgs)
        after_len = sum(len(m['content']) for m in out)
        self.assertLess(after_len, before_len)


if __name__ == '__main__':
    unittest.main()
