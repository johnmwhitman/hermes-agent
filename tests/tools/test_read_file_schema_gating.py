"""read_file schema diet (#95681): static unconditional format list
(anydoc bundled in core) + PDF-coverage teaching moved to the
response-time warning.

Maintainer-directed: the schema advertised anydoc-gated formats
unconditionally ("convert too when the optional anydoc converter is
available") and pre-taught the EXTRACTION COVERAGE WARNING's own
instructions. Now the format list renders only when anydoc is importable,
and the warning (read_extract.py) is the single teacher — it fires exactly
when pages are missing, with the page map and recovery commands.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))



class TestReadFileSchemaStatic(unittest.TestCase):
    """Gate DROPPED by maintainer decision: anydoc is a core dependency
    (bundled), so format support is stated unconditionally — a missing
    converter is a broken install handled by read_extract's teaching
    error, not a schema variant."""

    def test_formats_stated_unconditionally(self):
        from tools.file_tools import READ_FILE_SCHEMA

        desc = READ_FILE_SCHEMA["description"]
        for token in (".ipynb", ".docx", ".pptx", ".doc/.ppt/.xls",
                      "PDF (text layer)", "OpenDocument", "RTF", "EPUB"):
            self.assertIn(token, desc, token)
        # No availability hedging, no install mechanics, no gate.
        self.assertNotIn("when the optional", desc)
        self.assertNotIn("auto-installed", desc)
        self.assertNotIn("anydoc", desc)

    def test_pdf_wording_upgrades_with_hosted_ocr_route(self):
        """The ONE dynamic word: text-layer → scanned-or-text, keyed on
        hosted_ocr_available(). Nous gateway deliberately does not
        upgrade (Parse proxy live-probed broken 2026-08-28)."""
        import tools.file_tools as ft

        with patch("tools.read_extract.hosted_ocr_available",
                   return_value=True):
            d = ft._read_file_schema_overrides()["description"]
        self.assertIn("PDF (scanned or text)", d)
        self.assertNotIn("PDF (text layer)", d)
        with patch("tools.read_extract.hosted_ocr_available",
                   return_value=False):
            o = ft._read_file_schema_overrides()
        self.assertEqual(o, {})  # base wording stands

    def test_hosted_ocr_available_gate_states(self):
        """Hosted OCR needs both a direct key and literal config opt-in."""
        import tools.read_extract as rx

        # A direct key alone must never turn document upload on.
        with patch.dict(rx.os.environ, {"FIRECRAWL_API_KEY": "fc-x"}):
            with patch("hermes_cli.config.load_config_readonly_strict",
                       return_value={}):
                self.assertFalse(rx.hosted_ocr_available())
        # Explicit false keeps it off.
        with patch.dict(rx.os.environ, {"FIRECRAWL_API_KEY": "fc-x"}):
            with patch("hermes_cli.config.load_config_readonly_strict",
                       return_value={"file_tools": {"hosted_ocr": False}}):
                self.assertFalse(rx.hosted_ocr_available())
        # Only literal True opts in; truthy values are not policy.
        with patch.dict(rx.os.environ, {"FIRECRAWL_API_KEY": "fc-x"}):
            with patch("hermes_cli.config.load_config_readonly_strict",
                       return_value={"file_tools": {"hosted_ocr": "true"}}):
                self.assertFalse(rx.hosted_ocr_available())
        # Explicit true WITHOUT a key cannot provide a hosted route.
        with patch.dict(rx.os.environ, {}, clear=False):
            rx.os.environ.pop("FIRECRAWL_API_KEY", None)
            with patch("hermes_cli.config.load_config_readonly_strict",
                       return_value={"file_tools": {"hosted_ocr": True}}):
                self.assertFalse(rx.hosted_ocr_available())
        # Explicit true plus the direct key is the only enabled state.
        with patch.dict(rx.os.environ, {"FIRECRAWL_API_KEY": "fc-x"}):
            with patch("hermes_cli.config.load_config_readonly_strict",
                       return_value={"file_tools": {"hosted_ocr": True}}):
                self.assertTrue(rx.hosted_ocr_available())

    def test_hosted_ocr_config_load_failure_fails_closed(self):
        import tools.read_extract as rx

        with patch.dict(rx.os.environ, {"FIRECRAWL_API_KEY": "fc-x"}):
            with patch("hermes_cli.config.load_config_readonly_strict",
                       side_effect=OSError("unreadable config")):
                self.assertFalse(rx.hosted_ocr_available())
                self.assertEqual(
                    rx._hosted_ocr_config(),
                    (False, "fc-x", None),
                )

    def test_malformed_current_config_cannot_reuse_cached_opt_in(self):
        """A prior valid opt-in must not survive a broken current file."""
        from hermes_cli import config as config_mod
        from hermes_cli import managed_scope
        from tools import file_tools as ft
        from tools import read_extract as rx

        class NeedsOcrError(Exception):
            def __init__(self, pages):
                super().__init__("needs ocr")
                self.pages = pages

        calls = []

        class FakeAnydoc:
            @staticmethod
            def to_markdown(path, **kwargs):
                calls.append(kwargs)
                if not kwargs:
                    raise NeedsOcrError([1])
                return "HOSTED TEXT"

        FakeAnydoc.NeedsOcrError = NeedsOcrError

        with tempfile.TemporaryDirectory(prefix="hosted-ocr-config-") as temp:
            home = Path(temp)
            config_path = home / "config.yaml"
            scan_path = home / "scan.pdf"
            config_path.write_text(
                "file_tools:\n  hosted_ocr: true\n",
                encoding="utf-8",
            )
            scan_path.write_bytes(b"%PDF fixture")

            with patch.dict(
                rx.os.environ,
                {
                    "HERMES_HOME": str(home),
                    "FIRECRAWL_API_KEY": "fc-test",
                },
                clear=False,
            ):
                rx.os.environ.pop("HERMES_MANAGED_DIR", None)
                config_mod._LOAD_CONFIG_CACHE.clear()
                config_mod._LAST_EXPANDED_CONFIG_BY_PATH.clear()
                managed_scope.invalidate_managed_cache()
                try:
                    # Prime the ordinary loader's cache and last-known-good
                    # state from a real, valid current file.
                    self.assertTrue(
                        config_mod.load_config_readonly()["file_tools"]["hosted_ocr"]
                    )
                    self.assertTrue(rx.hosted_ocr_available())

                    # The ordinary config loader retains the previous value on
                    # this parse failure. The upload gate must independently
                    # verify the current merged-config read and fail closed.
                    config_path.write_text(
                        "file_tools:\n  hosted_ocr: [broken\n",
                        encoding="utf-8",
                    )

                    self.assertFalse(rx.hosted_ocr_available())
                    self.assertEqual(ft._read_file_schema_overrides(), {})
                    with patch.object(rx, "_anydoc", return_value=FakeAnydoc):
                        out = rx._extract_anydoc(str(scan_path))
                    self.assertIn("[NEEDS OCR", out)
                    self.assertEqual(calls, [{}])
                finally:
                    config_mod._LOAD_CONFIG_CACHE.clear()
                    config_mod._LAST_EXPANDED_CONFIG_BY_PATH.clear()
                    managed_scope.invalidate_managed_cache()

    def test_strict_current_read_preserves_managed_precedence(self):
        from hermes_cli import config as config_mod
        from hermes_cli import managed_scope

        with tempfile.TemporaryDirectory(prefix="hosted-ocr-precedence-") as temp:
            root = Path(temp)
            home = root / "profile"
            managed = root / "managed"
            home.mkdir()
            managed.mkdir()
            (home / "config.yaml").write_text(
                "file_tools:\n  hosted_ocr: true\n",
                encoding="utf-8",
            )
            (managed / "config.yaml").write_text(
                "file_tools:\n  hosted_ocr: false\n",
                encoding="utf-8",
            )

            with patch.dict(
                os.environ,
                {
                    "HERMES_HOME": str(home),
                    "HERMES_MANAGED_DIR": str(managed),
                },
                clear=False,
            ):
                config_mod._LOAD_CONFIG_CACHE.clear()
                config_mod._LAST_EXPANDED_CONFIG_BY_PATH.clear()
                managed_scope.invalidate_managed_cache()
                try:
                    cfg = config_mod.load_config_readonly_strict()
                    self.assertIs(
                        cfg["file_tools"]["hosted_ocr"],
                        False,
                    )
                finally:
                    config_mod._LOAD_CONFIG_CACHE.clear()
                    config_mod._LAST_EXPANDED_CONFIG_BY_PATH.clear()
                    managed_scope.invalidate_managed_cache()

    def test_runtime_route_is_direct_key_only(self):
        """Runtime route requires explicit opt-in and never uses Nous."""
        import tools.read_extract as rx

        with patch.dict(rx.os.environ, {"FIRECRAWL_API_KEY": "fc-x"}):
            with patch("hermes_cli.config.load_config_readonly_strict",
                       return_value={}):
                enabled, key, url = rx._hosted_ocr_config()
        self.assertFalse(enabled)
        self.assertEqual(key, "fc-x")
        self.assertIsNone(url)
        with patch.dict(rx.os.environ, {"FIRECRAWL_API_KEY": "fc-x"}):
            with patch("hermes_cli.config.load_config_readonly_strict",
                       return_value={"file_tools": {"hosted_ocr": True}}):
                enabled, key, url = rx._hosted_ocr_config()
        self.assertTrue(enabled)
        self.assertEqual(key, "fc-x")
        self.assertIsNone(url)
        with patch("hermes_cli.config.load_config_readonly_strict",
                   return_value={}):
            rx.os.environ.pop("FIRECRAWL_API_KEY", None)
            enabled, key, url = rx._hosted_ocr_config()
        self.assertFalse(enabled)
        self.assertIsNone(key)
        self.assertIsNone(url)

    def test_coverage_warning_teaching_left_to_the_warning(self):
        """The response-time warning owns the recovery curriculum."""
        from tools.file_tools import READ_FILE_SCHEMA

        desc = READ_FILE_SCHEMA["description"]
        self.assertNotIn("EXTRACTION COVERAGE WARNING", desc)
        self.assertNotIn("NEEDS OCR", desc)
        self.assertNotIn("pdftoppm", desc)
        import inspect
        from tools import read_extract

        src = inspect.getsource(read_extract)
        self.assertIn("NEEDS OCR", src)
        self.assertIn("pdftoppm", src)
        self.assertIn("vision_analyze", src)

    def test_binary_note_stays_last(self):
        from tools.file_tools import READ_FILE_SCHEMA

        desc = READ_FILE_SCHEMA["description"]
        self.assertLess(desc.find("EPUB"), desc.find("Cannot read images/binary"))

    def test_missing_anydoc_error_teaches_install(self):
        from tools.read_extract import _anydoc_missing_error

        err = _anydoc_missing_error("x.epub")
        self.assertIn("firecrawl-anydoc", err)
        self.assertNotEqual(err, "Unsupported document type: 'x.epub'")


class TestNeedsOcrPath(unittest.TestCase):
    """anydoc>=0.2 NeedsOcrError wiring: hosted OCR attempt + typed warning
    (maintainer caveats: #1 nous-gateway Parse was live-probed HTTP 500 →
    attempt-and-fall-through; #2 warning recommends LOCAL OCR skills)."""

    def _fake_mod(self, hosted_result=None, hosted_exc=None):
        class NeedsOcrError(Exception):
            def __init__(self, pages):
                super().__init__("needs ocr")
                self.pages = pages

        calls = []

        class Mod:
            pass

        mod = Mod()
        mod.NeedsOcrError = NeedsOcrError

        def to_markdown(path, **kw):
            calls.append(kw)
            if not kw:
                raise NeedsOcrError([2, 3])
            if hosted_exc is not None:
                raise hosted_exc
            return hosted_result

        mod.to_markdown = to_markdown
        return mod, calls

    def test_hosted_success_returns_ocr_text(self):
        from tools import read_extract as rx

        mod, calls = self._fake_mod(hosted_result="OCR TEXT")
        with patch.object(rx, "_anydoc", return_value=mod),              patch.object(rx, "_hosted_ocr_config",
                          return_value=(True, "key", None)),              patch.object(rx.os.path, "getsize", return_value=10):
            out = rx._extract_anydoc("scan.pdf")
        self.assertEqual(out, "OCR TEXT\n")
        self.assertEqual(calls[1].get("ocr"), "hosted")

    def test_hosted_failure_warns_and_prefers_local_skills(self):
        from tools import read_extract as rx

        mod, _ = self._fake_mod(hosted_exc=RuntimeError("HTTP 500"))
        with patch.object(rx, "_anydoc", return_value=mod),              patch.object(rx, "_hosted_ocr_config",
                          return_value=(True, "key", "https://gw")),              patch.object(rx.os.path, "getsize", return_value=10):
            out = rx._extract_anydoc("scan.pdf")
        self.assertIn("[NEEDS OCR", out)
        self.assertIn("pages 2, 3", out)
        self.assertIn("attempted and failed", out)
        # Maintainer-directed: HINT at checking for an OCR skill; never
        # name one (none is guaranteed to exist), never sell config knobs.
        self.assertIn("check whether an OCR skill is available", out)
        self.assertIn("skills_list", out)
        self.assertNotIn("ocr-and-documents", out)
        self.assertNotIn("marker-pdf", out)
        self.assertNotIn("hosted_ocr", out)

    def test_disabled_warns_without_attempt(self):
        from tools import read_extract as rx

        mod, calls = self._fake_mod()
        with patch.object(rx, "_anydoc", return_value=mod),              patch.object(rx, "_hosted_ocr_config",
                          return_value=(False, None, None)),              patch.object(rx.os.path, "getsize", return_value=10):
            out = rx._extract_anydoc("scan.pdf")
        self.assertIn("[NEEDS OCR", out)
        self.assertEqual(len(calls), 1)  # no hosted attempt
        # Same shape when disabled: skill hint, no knob advertising.
        self.assertIn("check whether an OCR skill is available", out)
        self.assertNotIn("hosted_ocr", out)
        self.assertNotIn("ocr-and-documents", out)

    def test_key_without_explicit_opt_in_never_attempts_hosted_ocr(self):
        from tools import read_extract as rx

        mod, calls = self._fake_mod()
        with patch.object(rx, "_anydoc", return_value=mod), \
             patch.object(rx.os.path, "getsize", return_value=10), \
             patch.dict(rx.os.environ, {"FIRECRAWL_API_KEY": "fc-x"}), \
             patch("hermes_cli.config.load_config_readonly_strict", return_value={}):
            out = rx._extract_anydoc("scan.pdf")
        self.assertIn("[NEEDS OCR", out)
        self.assertEqual(calls, [{}])

    def test_explicit_opt_in_attempts_hosted_ocr(self):
        from tools import read_extract as rx

        mod, calls = self._fake_mod(hosted_result="OCR TEXT")
        with patch.object(rx, "_anydoc", return_value=mod), \
             patch.object(rx.os.path, "getsize", return_value=10), \
             patch.dict(rx.os.environ, {"FIRECRAWL_API_KEY": "fc-x"}), \
             patch("hermes_cli.config.load_config_readonly_strict",
                   return_value={"file_tools": {"hosted_ocr": True}}):
            out = rx._extract_anydoc("scan.pdf")
        self.assertEqual(out, "OCR TEXT\n")
        self.assertEqual(calls, [{}, {"ocr": "hosted", "api_key": "fc-x"}])

    def test_pin_lockstep(self):
        """pyproject core pin and lazy_deps self-heal pin must match."""
        import re
        from pathlib import Path

        py = Path("pyproject.toml").read_text(encoding="utf-8")
        lz = Path("tools/lazy_deps.py").read_text(encoding="utf-8")
        m1 = re.search(r'"firecrawl-anydoc==([\d.]+)"', py)
        m2 = re.search(r'"firecrawl-anydoc==([\d.]+)"', lz)
        self.assertIsNotNone(m1)
        self.assertIsNotNone(m2)
        self.assertEqual(m1.group(1), m2.group(1))


if __name__ == "__main__":
    unittest.main()
