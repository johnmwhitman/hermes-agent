# tests/plugins/context_engine/test_engine_ccr_integration.py

import json
import tempfile
import unittest
from pathlib import Path

from plugins.context_engine.ast_code_compressor import AstCodeCompressor
from plugins.context_engine.ccr_store import CcrStore
from plugins.context_engine.toon_compressor import ToonCompressor


class AstCcrIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.store = CcrStore(self.tmpdir / "ccr.db")
        self.eng = AstCodeCompressor()
        self.eng.set_ccr_store(self.store)

    def _make_messages(self, big_py):
        return [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "small"},
            {"role": "assistant", "content": "small"},
            {"role": "user", "content": "small"},
            {"role": "user", "content": f"```python\n{big_py}\n```"},
        ]

    def test_compress_stores_original(self):
        big_py = "import os\n" + ("def f(x):\n    return x*2\n" * 200) + "\n"
        out = self.eng.compress(self._make_messages(big_py))
        skeleton = out[4]["content"]
        # Extract hash from skeleton header
        import re
        m = re.search(r"original_sha256=([0-9a-f]{16})", skeleton)
        self.assertIsNotNone(m)
        hash_short = m.group(1)
        # Verify CCR has the body that matches the engine's hash
        rec = self.store.get(hash_short)
        self.assertIsNotNone(rec)
        # The engine stored the regex-captured body (with leading newline after
        # the opening fence); the round-trip is correct because the engine
        # hashes the same body it stored.
        self.assertTrue(rec["content"].decode("utf-8").startswith("import os\n"))

    def test_skeleton_retrievable_via_tool(self):
        from plugins.context_engine.ccr_store import ccr_retrieve
        big_py = "import os\n" + ("def f(x):\n    return x*2\n" * 200) + "\n"
        out = self.eng.compress(self._make_messages(big_py))
        skeleton = out[4]["content"]
        import re
        m = re.search(r"original_sha256=([0-9a-f]{16})", skeleton)
        result = ccr_retrieve(m.group(1), profile_dir=self.tmpdir)
        self.assertNotIn("error", result)
        self.assertEqual(result["content_type"], "python")
        self.assertIn("import os", result["content"])

    def test_compression_still_works_without_ccr(self):
        # If no CCR is wired, compression should still produce skeletons,
        # just without the reversibility benefit.
        eng = AstCodeCompressor()  # no set_ccr_store
        big_py = "import os\n" + ("def f(x):\n    return x*2\n" * 200) + "\n"
        out = eng.compress(self._make_messages(big_py))
        skeleton = out[4]["content"]
        self.assertIn("AST skeleton", skeleton)


class ToonCcrIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.store = CcrStore(self.tmpdir / "ccr.db")
        self.eng = ToonCompressor()
        self.eng.set_ccr_store(self.store)

    def test_compress_stores_original(self):
        big_json = json.dumps([
            {"id": i, "name": f"item-{i}", "tokens": i * 100}
            for i in range(50)
        ])
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "small"},
            {"role": "assistant", "content": "small"},
            {"role": "user", "content": "small"},
            {"role": "user", "content": big_json},
        ]
        out = self.eng.compress(msgs)
        compressed = out[4]["content"]
        import re
        m = re.search(r"original_sha256=([0-9a-f]{16})", compressed)
        self.assertIsNotNone(m)
        hash_short = m.group(1)
        rec = self.store.get(hash_short)
        self.assertIsNotNone(rec)
        # The engine stored the regex-captured blob; round-trip is internally consistent.
        self.assertEqual(rec["content"].decode("utf-8"), big_json)


if __name__ == "__main__":
    unittest.main()
