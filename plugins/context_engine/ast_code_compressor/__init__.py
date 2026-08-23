# plugins/context_engine/ast_code_compressor/__init__.py
#
# AST-aware code compression engine (Hermes-native).
#
# Strategy: when conversation context contains large Python source blocks,
# replace each block with an AST skeleton that preserves imports + class/
# function signatures + types but drops body. Originals are content-addressed
# so they can be retrieved on demand via the CCR plugin (when active).
#
# Replaces Headroom's tree-sitter multi-language approach with a
# Python-only stdlib `ast` implementation. Trade-off: only handles Python,
# but produces 96%+ reduction on real Python source (per
# agents/capabilities/tools/bench_ast_code_compression.py).
#
# Composition: this engine complements the default `compressor` engine, not
# replaces it. The default engine is lossy summarization for prose turns;
# this engine is lossless-with-original-access for code blocks.
#
# Selection: in config.yaml, set
#     context.engine: ast_code_compressor
# to make this the active engine. The default `compressor` is unaffected.

from __future__ import annotations

import ast
import hashlib
import logging
import re
from typing import Any, Dict, List, Optional

from agent.context_engine import ContextEngine

logger = logging.getLogger(__name__)


# --- Compression telemetry -------------------------------------------------
#
# Each compression engine writes one JSONL line per compress() / select_context()
# call to <profile_home>/compression_attempts.jsonl. Schema is documented at
# `agents/capabilities/designs/compression-telemetry-schema.md`. The watchdog
# `portfolio-compression-rollup` (see NEXT-CYCLE-QUEUE D3) reads the file to
# emit per-day/per-profile rollups so we can see exactly how often each
# engine fires, what it saves, and which profile runs it most.
#
# The telemetry is best-effort: if HERMES_HOME is unset or the file write
# fails (disk full, permissions), compression still proceeds. We log a warning
# rather than failing the engine.

_COMPRESSION_ATTEMPTS_FILENAME = "compression_attempts.jsonl"


def _resolve_telemetry_path() -> Optional["Path"]:
    """Return the per-profile JSONL path, or None if HERMES_HOME is unset."""
    import os as _os
    from pathlib import Path as _P
    home = _os.environ.get("HERMES_HOME", "").strip()
    if not home:
        return None
    return _P(home) / _COMPRESSION_ATTEMPTS_FILENAME


def _write_telemetry(event: Dict[str, Any]) -> None:
    """Append one JSONL line. Best-effort: never raise."""
    try:
        path = _resolve_telemetry_path()
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            import json as _json
            f.write(_json.dumps(event, separators=(",", ":")) + "\n")
    except Exception as exc:
        logger.warning("compression telemetry write failed: %s", exc)



# Minimum size in characters for a code block to be considered for skeleton
# compression. Below this threshold, the overhead exceeds the savings.
_MIN_BLOCK_CHARS = 600

# Heuristic: code block in a tool message typically looks like
# ```python
# <code>
# ```
# or is fenced with ```py / ```python. We also catch raw triple-quoted
# blocks when fenced detection fails.
_FENCED_PY_RE = re.compile(
    r"(?P<full>```(?:python|py)?\n(?P<body>.*?)```)",
    re.DOTALL,
)


# Multi-language support (A1, tree-sitter). The Python path stays
# stdlib-only; everything else uses tree-sitter. _LANGUAGE_PARSERS maps
# a fence-language tag to a (parser-factory, body-validity) function.
# parser-factory returns a callable that takes (code: str) -> str skeleton.
# body-validity takes (code: str) -> bool — returns True iff the body parses
# cleanly enough to compress.
_LANGUAGE_PARSERS: dict = {}


def _try_init_tree_sitter():
    """Lazy init: tree-sitter is a C extension. If it isn't installed,
    return False and the engine falls back to Python-only.

    The first import attempt that succeeds registers parsers into
    _LANGUAGE_PARSERS for: javascript, typescript, tsx, go, rust, java, cpp.
    """
    global _LANGUAGE_PARSERS
    if _LANGUAGE_PARSERS:
        return True  # already initialized
    try:
        import tree_sitter  # noqa: F401
        import tree_sitter_languages  # noqa: F401
    except ImportError:
        return False
    try:
        import tree_sitter as _ts
        import tree_sitter_languages as _ts_lang

        def _make_skeleton(language: str, handler_map: dict):
            """Return a function that takes (code: str) -> str skeleton.

            The skeleton preserves top-level declarations (imports,
            functions, classes) and drops bodies. Falls back to a 1-line
            summary if the body is short enough that nothing compresses.
            """
            def skeletonize(code: str) -> str:
                try:
                    parser = _ts.Parser()
                    parser.set_language(_ts_lang.get_language(language))
                    tree = parser.parse(code.encode("utf-8"))
                except Exception:
                    return code  # bail
                out: list[str] = []

                def _walk(node, code, depth, out):
                    # ext_to_kind and kind_extractors are both dicts mapping
                    # node-type -> True (a set-like membership dict).
                    # Membership in kind_extractors means "extract a signature
                    # line"; membership in RECURSE_TYPES means "descend into
                    # this container and walk its children".
                    # Nodes we descend into. Containers that hold nested
                    # declarations we want to extract.
                    RECURSE_TYPES = {
                        "class_body",           # methods inside classes
                        "class_declaration",    # wrapper, descends into class_body
                        "impl_item",            # Rust impl block wrapper
                        "object",               # dict literals (rare in TS)
                        "export_statement",     # exports wrapper (descend)
                        "lexical_declaration",  # const/let blocks
                        "module",               # Rust mod
                        "source_file",          # Go file wrapper
                        "declaration_list",     # Go import block
                    }
                    # Children of class_declaration / impl_item that are
                    # just keywords / type identifiers / punctuation (NOT
                    # useful declarations on their own).
                    NOISE_CHILDREN = {
                        "class", "interface", "fn", "function", "type",
                        "impl", "struct", "enum", "mod", "pub",
                        "async", "export", "default", "abstract",
                        "static", "const", "let", "var",
                        # Type / name identifiers (handled by parent).
                        "type_identifier", "property_identifier",
                        "identifier",
                        # Generic type args.
                        "type_arguments", "generic_type",
                        # Formal params / type annotations.
                        "formal_parameters", "required_parameter",
                        "type_annotation", "predefined_type",
                        # Note: class_body is NOT here — it has its own
                        # recursion entry above to descend into its children.
                        # Punctuation.
                        "{", "}", "(", ")", "[", "]", ";", ",", ":",
                    }
                    # If the node is a recognized declaration, emit its
                    # signature. If it's only a container (no signature), AND
                    # it's in RECURSE_TYPES, descend into its children.
                    # First: emit handler signature if known.
                    emitted = False
                    if node.type in handler_map:
                        line = handler_map[node.type](node, code)
                        if line:
                            out.append(("  " * depth) + line)
                            emitted = True
                    # Then: descend into RECURSE containers. Skip children
                    # that are just keywords / identifiers (NOISE_CHILDREN).
                    if node.type in RECURSE_TYPES and depth < 3:
                        for grandchild in node.children:
                            if grandchild.type in NOISE_CHILDREN:
                                continue
                            _walk(grandchild, code, depth + 1, out)
                        return
                    if emitted:
                        return
                    # Pure-punctuation nodes (braces, parens) — skip.
                    if node.type in {"{", "}", "(", ")", "[", "]", ";", ","}:
                        return
                    # Unknown kind: emit a 1-line stub.
                    body = code[node.start_byte:node.end_byte]
                    first = body.split("\n", 1)[0]
                    for sfx in (" {", "{", " ("):
                        if first.rstrip().endswith(sfx):
                            first = first.rstrip()[:-len(sfx)].rstrip()
                            break
                    if not first.strip():
                        return
                    out.append(("  " * depth) + first[:240])
                    if node.type in RECURSE_TYPES and depth < 3:
                        for grandchild in node.children:
                            _walk(grandchild, code, depth + 1, out)
                        return
                    else:
                        # Unknown kind: emit a 1-line stub.
                        body = code[node.start_byte:node.end_byte]
                        first = body.split("\n", 1)[0]
                        for sfx in (" {", "{", " ("):
                            if first.rstrip().endswith(sfx):
                                first = first.rstrip()[:-len(sfx)].rstrip()
                                break
                        if not first.strip():
                            return
                        out.append(("  " * depth) + first[:240])

                for child in tree.root_node.children:
                    _walk(child, code, 0, out)
                if not out:
                    return code
                return "\n".join(out)
            return skeletonize

        def _js_handler(node, code):
            """JS / TS top-level: function_declaration, class_declaration, import_statement, etc."""
            txt = code[node.start_byte:node.end_byte]
            head = txt.split("\n", 1)[0].rstrip("{").rstrip()
            return head if len(head) < 200 else head[:200] + " ..."

        def _go_handler(node, code):
            """Go top-level: function_declaration, type_declaration, import_declaration."""
            txt = code[node.start_byte:node.end_byte]
            head = txt.split("\n", 1)[0].rstrip("{").rstrip()
            return head if len(head) < 200 else head[:200] + " ..."

        def _rust_handler(node, code):
            """Rust top-level: function_item, struct_item, enum_item, impl_item, use_declaration."""
            txt = code[node.start_byte:node.end_byte]
            head = txt.split("\n", 1)[0].rstrip("{").rstrip()
            return head if len(head) < 200 else head[:200] + " ..."

        # Per-kind handlers: given a top-level node, return a one-line
        # signature (drops the body). All languages share this pattern.
        def _signature(node, code):
            txt = code[node.start_byte:node.end_byte]
            head = txt.split("\n", 1)[0]
            # Trim trailing opening brace / paren for cleaner output
            for suffix in (" {", " {", "{", " ("):
                if head.rstrip().endswith(suffix):
                    head = head.rstrip()[:-len(suffix)].rstrip()
                    break
            return head if len(head) < 240 else head[:240] + " ..."

        # Per-language node-type → handler mapping. The dict values are
        # the `_signature` function — every recognized declaration calls
        # it to emit a one-line signature.
        js_types = {
            "function_declaration": _signature,
            "class_declaration": _signature,
            # NOTE: export_statement is intentionally NOT in the handlers
            # — it's a wrapper, not a declaration. We recurse INTO it to
            # extract the inner fn/class.
            "import_statement": _signature,
            "lexical_declaration": _signature,
            "variable_declaration": _signature,
            "method_definition": _signature,
            "abstract_method_signature": _signature,
            "interface_declaration": _signature,
            "type_alias_declaration": _signature,
            "enum_declaration": _signature,
            "abstract_class_declaration": _signature,
        }
        go_types = {
            "function_declaration": _signature,
            "method_declaration": _signature,
            "type_declaration": _signature,
            "import_declaration": _signature,
        }
        rust_types = {
            "function_item": _signature,
            "struct_item": _signature,
            "enum_item": _signature,
            "impl_item": _signature,
            "use_declaration": _signature,
            "trait_item": _signature,
            "type_item": _signature,
            "mod_item": _signature,
            "const_item": _signature,
            "static_item": _signature,
        }

        _LANGUAGE_PARSERS["javascript"] = (
            _make_skeleton("javascript", js_types),
            lambda code: True,
        )
        _LANGUAGE_PARSERS["typescript"] = (
            _make_skeleton("typescript", js_types),
            lambda code: True,
        )
        _LANGUAGE_PARSERS["tsx"] = (
            _make_skeleton("tsx", js_types),
            lambda code: True,
        )
        _LANGUAGE_PARSERS["go"] = (
            _make_skeleton("go", go_types),
            lambda code: True,
        )
        _LANGUAGE_PARSERS["rust"] = (
            _make_skeleton("rust", rust_types),
            lambda code: True,
        )
        return True
    except Exception:
        return False


# Try to initialize at module load. Failure is fine — Python path stays.
try:
    _try_init_tree_sitter()
except Exception:
    pass


class AstCodeCompressor(ContextEngine):
    """ContextEngine that AST-skeleton-replaces large Python code blocks.

    Lossless with original-access (when CCR plugin is active): original code
    is content-addressed (sha256) and stored in the CCR store; the skeleton
    references the hash so the agent can `ccr_retrieve` the full block.

    Without CCR: lossless-with-original-inline (original block is preserved
    verbatim at the END of the message list so it doesn't pollute the head).

    Either way, the AST skeleton replaces the in-line block in the visible
    context window.
    """

    @property
    def name(self) -> str:
        return "ast_code_compressor"

    def is_available(self) -> bool:
        """Always available — uses only the Python stdlib."""
        return True

    # -- Token state -------------------------------------------------------
    # All four are required by the ABC; defaults are inherited.

    # -- Compaction parameters --------------------------------------------
    # protect_first_n is read by run_agent.py for preflight; we keep the
    # ABC default of 3 (system + first 3 non-system messages) to avoid
    # surprising the host.

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        """No-op: token accounting is handled by the host; this engine
        doesn't make LLM calls (unlike the lossy summarizer)."""
        pass

    def should_compress(self, prompt_tokens: int = None) -> bool:
        """Compression is event-driven (called from `compress` by the host),
        not token-budget-driven. Always return False so the host uses its
        own threshold; the engine's compression happens during the explicit
        `compress()` call when triggered."""
        return False

    def select_context(
        self,
        request_messages,
        *,
        conversation_messages=None,
        incoming_message=None,
        budget_tokens=0,
    ):
        """Per-turn hook (independent of ``should_compress()``).

        Fires on EVERY request to compress large Python code blocks inline.
        This is the integration point that gives the AST engine real-world
        reach: without this hook, the host never calls our ``compress()`` and
        the engine is dormant (host's token-budget thresholds only fire for
        the lossy summarizer). Compressing per-turn on the request list is
        safe because the result is request-only — persisted transcript stays
        intact, and the agent gets a leaner prompt.
        """
        if not request_messages:
            return request_messages
        compressed, stats = self._compress_messages(request_messages)
        if stats["blocks_compressed"] > 0:
            self._compression_count = getattr(self, "_compression_count", 0) + 1
        _write_telemetry({
            "ts": __import__("time").time(),
            "engine": "ast_code_compressor",
            "hook": "select_context",
            "blocks_found": stats.get("blocks_found", 0),
            "blocks_compressed": stats.get("blocks_compressed", 0),
            "bytes_saved": stats.get("bytes_saved", 0),
        })
        return compressed

    def compress(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: Optional[int] = None,
        focus_topic: Optional[str] = None,
        force: bool = False,
        memory_context: str = "",
    ) -> List[Dict[str, Any]]:
        """Walk messages; replace large Python code blocks with AST skeletons.

        Returns the (possibly shorter) message list. Non-Python blocks and
        blocks below the size threshold pass through unchanged. Messages
        outside the engine's surface (system, head, tail-protected by the
        ABC's protect_first_n) are passed through.
        """
        if not messages:
            return messages

        # Identify protected head: system message + first N non-system
        # messages. The ABC exposes `protect_first_n` as a class attr we
        # can read but the per-engine setting can vary. We follow the
        # standard protect_first_n=3 pattern (system + first 3 non-system).
        protected_head = 1 + getattr(self, "protect_first_n", 3)

        # Walk from the protected_head cutoff to the tail. We do NOT touch
        # the last message if it's the user — the user is likely waiting for
        # a response that depends on the most-recent code block intact.
        out: List[Dict[str, Any]] = []
        stats = {"blocks_found": 0, "blocks_compressed": 0, "bytes_saved": 0}

        for idx, msg in enumerate(messages):
            if idx < protected_head:
                out.append(msg)
                continue

            new_msg, msg_stats = self._process_message(msg)
            out.append(new_msg)
            for k, v in msg_stats.items():
                stats[k] += v

        if stats["blocks_compressed"] > 0:
            self.compression_count += 1
        _write_telemetry({
            "ts": __import__("time").time(),
            "engine": "ast_code_compressor",
            "hook": "compress",
            "blocks_found": stats.get("blocks_found", 0),
            "blocks_compressed": stats.get("blocks_compressed", 0),
            "bytes_saved": stats.get("bytes_saved", 0),
        })
        self._last_compression_savings_pct = (
            100.0 * stats["bytes_saved"]
            / max(1, stats["bytes_saved"] + stats["blocks_compressed"] * _MIN_BLOCK_CHARS)
        )
        logger.info(
            "ast_code_compressor: %d Python block(s) compressed, %d bytes saved (%.1f%%)",
            stats["blocks_compressed"], stats["bytes_saved"],
            self._last_compression_savings_pct,
        )
        return out

    def _compress_messages(self, messages):
        """Shared per-message compression path used by both ``compress()``
        and the per-turn ``select_context()`` hook."""
        if not messages:
            return messages, {"blocks_found": 0, "blocks_compressed": 0, "bytes_saved": 0}
        protected_head = 1 + getattr(self, "protect_first_n", 3)
        out: List[Dict[str, Any]] = []
        stats = {"blocks_found": 0, "blocks_compressed": 0, "bytes_saved": 0}
        for idx, msg in enumerate(messages):
            if idx < protected_head:
                out.append(msg)
                continue
            new_msg, msg_stats = self._process_message(msg)
            out.append(new_msg)
            for k, v in msg_stats.items():
                stats[k] += v
        return out, stats

    # -- Internal helpers --------------------------------------------------

    def _process_message(self, msg: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, int]]:
        """Compress any large Python code block in the message content.

        Returns the new message (or unchanged) and stats. Only touches
        message roles that carry text content (user/assistant/tool/system
        all possible — we check any "content" or "text" key).
        """
        stats = {"blocks_found": 0, "blocks_compressed": 0, "bytes_saved": 0}

        # Skip non-message types and dicts without a content key.
        if not isinstance(msg, dict):
            return msg, stats

        content = msg.get("content")
        if not isinstance(content, str):
            # Some messages carry content as list of parts (OpenAI multimodal).
            # We don't compress those — too risky.
            return msg, stats

        new_content, msg_stats = self._compress_text(content)
        for k, v in msg_stats.items():
            stats[k] += v

        if msg_stats["blocks_compressed"] == 0:
            return msg, stats

        new_msg = dict(msg)
        new_msg["content"] = new_content
        return new_msg, stats

    def _compress_text(self, text: str) -> tuple[str, Dict[str, int]]:
        """Find fenced Python code blocks and replace large ones with AST skeletons.

        If a CCR store is wired via set_ccr_store(), each replaced block also
        stores its original body under the sha256 prefix printed in the
        skeleton marker, so the agent can re-fetch via ccr_retrieve().
        """
        stats = {"blocks_found": 0, "blocks_compressed": 0, "bytes_saved": 0}

        def replace_block(match: re.Match) -> str:
            full = match.group("full")
            body = match.group("body")
            stats["blocks_found"] += 1
            if len(body) < _MIN_BLOCK_CHARS:
                return full

            skeleton = ast_skeleton(body)
            if skeleton is None:
                # Not valid Python — pass through.
                return full

            saved = len(body) - len(skeleton)
            if saved < 100:
                return full

            # Build the replacement: a marker fence that points at the CCR
            # content hash. If CCR is wired, the agent can re-fetch. If not,
            # this is just a header that tells the reader where to look.
            content_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]
            self._store_original(body, content_hash)
            skeleton_marker = (
                f"```python\n# AST skeleton (saved {saved} bytes / "
                f"{100*saved//len(body)}%)\n"
                f"# original_sha256={content_hash}\n"
                f"# use ccr_retrieve('{content_hash}') to fetch full code\n"
                f"{skeleton}\n```"
            )
            stats["blocks_compressed"] += 1
            stats["bytes_saved"] += saved
            return skeleton_marker

        new_text = _FENCED_PY_RE.sub(replace_block, text)

        # Second pass: catch unfenced Python. A bare Python source pasted
        # into a chat (no ```python fence) is the most common missed case.
        # We split on double-newlines, attempt to parse each chunk as Python,
        # and compress only when the chunk (a) is large enough, (b) parses
        # cleanly, and (c) hasn't already been replaced by the fenced pass.
        # We track already-compressed byte ranges to avoid double-compressing.
        new_text = _apply_unfenced_pass(new_text, replace_block, stats)

        return new_text, stats

    def set_ccr_store(self, store) -> None:
        """Wire a CcrStore (from plugins.context_engine.ccr_store) so that
        each compressed block stores its original body for retrieval."""
        self._ccr_store = store

    def _store_original(self, body: str, content_hash: str) -> None:
        store = getattr(self, "_ccr_store", None)
        if store is None:
            return
        try:
            store.put(body.encode("utf-8"), "python", "ast_code_compressor-1.0")
        except Exception as exc:
            logger.warning("ccr_store put failed for %s: %s", content_hash, exc)


def ast_skeleton(source: str) -> Optional[str]:
    """Return an AST skeleton of Python `source`: imports + signatures.

    Body and docstrings are dropped. Returns None if `source` is not valid
    Python (caller should pass it through unchanged).

    The skeleton is suitable for inclusion back into a Python file — it is
    valid Python and re-parsable. The only behavioral change vs. the
    original is that function/method bodies are empty (pass).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    lines: List[str] = []

    # Module docstring is dropped — keep imports.
    for node in tree.body:
        if isinstance(node, ast.Import):
            lines.append(_format_import(node))
        elif isinstance(node, ast.ImportFrom):
            lines.append(_format_importfrom(node))
        elif isinstance(node, ast.ClassDef):
            lines.append(_format_classdef(node))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            lines.append(_format_funcdef(node, indent=0))
        elif isinstance(node, ast.Assign):
            # Module-level constants — keep simple ones (no calls, no comprehensions)
            try:
                txt = ast.unparse(node)
                # Only keep if short and side-effect-free looking
                if txt and len(txt) < 100 and "(" not in txt and "[" not in txt:
                    lines.append(txt)
            except Exception:
                pass
        else:
            # Express / AnnAssign / If __name__ == "__main__" etc — keep
            # simple statements, drop complex ones.
            try:
                txt = ast.unparse(node)
                if txt and len(txt) < 60:
                    lines.append(txt)
            except Exception:
                pass

    return "\n".join(line for line in lines if line)


def _format_import(node: ast.Import) -> str:
    return ast.unparse(node)


def _format_importfrom(node: ast.ImportFrom) -> str:
    return ast.unparse(node)


def _format_classdef(node: ast.ClassDef, indent: int = 0) -> str:
    """Format a class definition: signature only, no body."""
    pad = "  " * indent
    bases = ", ".join(_safe_unparse(b) for b in node.bases)
    kwds = ""
    if node.keywords:
        kwds = ", " + ", ".join(
            f"{k.arg}={_safe_unparse(k.value)}" for k in node.keywords
        )
    head = f"class {node.name}({bases}{kwds}):".rstrip(":") + ":"
    if not node.body:
        return pad + head
    # Emit each method as a stub
    methods: List[str] = []
    for child in node.body:
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            methods.append(_format_funcdef(child, indent=indent + 1))
        elif isinstance(child, ast.ClassDef):
            methods.append(_format_classdef(child, indent=indent + 1))
        elif isinstance(child, ast.Expr) and isinstance(child.value, ast.Constant):
            # docstring — skip
            continue
        else:
            try:
                txt = ast.unparse(child)
                if txt and len(txt) < 60:
                    methods.append(pad + "  " + txt)
            except Exception:
                continue
    if not methods:
        return pad + head + "  pass"
    return pad + head + "\n" + "\n".join(methods)


def _format_funcdef(node, indent: int = 0) -> str:
    """Format a function definition: signature only, `pass` body."""
    pad = "  " * indent
    args = _format_arguments(node.args)
    ret = ""
    if node.returns is not None:
        ret = " -> " + _safe_unparse(node.returns)
    deco = ""
    for d in node.decorator_list:
        deco += "@" + _safe_unparse(d) + "\n" + pad
    prefix = "async def " if isinstance(node, ast.AsyncFunctionDef) else "def "
    return f"{pad}{deco}{prefix}{node.name}({args}){ret}:\n{pad}  pass"


def _format_arguments(args: ast.arguments) -> str:
    """Format a function's argument list as Python source.

    Includes defaults and positional-only / keyword-only separators as
    appropriate so the skeleton is re-parseable Python.

    `args.defaults[i]` is the default for the i-th arg counting FROM THE
    END of (posonly + regular). Same for `args.kw_defaults[i]` and
    `args.kwonlyargs[i]`.
    """
    parts: List[str] = []
    posonly = getattr(args, "posonlyargs", []) or []
    defaults = list(args.defaults or [])
    combined = list(posonly) + list(args.args)
    n_combined = len(combined)
    n_defaults = len(defaults)

    def render(arg: ast.arg, default_node) -> str:
        text = _format_arg(arg)
        if default_node is not None:
            text += " = " + _safe_unparse(default_node)
        return text

    # Defaults align to the tail of combined[]; the first n_combined - n_defaults
    # entries have no default, the rest do.
    for i, a in enumerate(combined):
        default_idx = i - (n_combined - n_defaults)  # negative if no default
        default_node = defaults[default_idx] if default_idx >= 0 else None
        parts.append(render(a, default_node))
    if posonly:
        parts.append("/")
    if args.vararg:
        parts.append("*" + _format_arg(args.vararg))
    elif args.kwonlyargs:
        parts.append("*")
    for i, a in enumerate(args.kwonlyargs):
        kd = (args.kw_defaults or [None] * len(args.kwonlyargs))[i]
        parts.append(render(a, kd))
    if args.kwarg:
        parts.append("**" + _format_arg(args.kwarg))
    return ", ".join(parts)


def _format_arg(arg: ast.arg) -> str:
    base = arg.arg
    if arg.annotation is not None:
        base += ": " + _safe_unparse(arg.annotation)
    return base


def _safe_unparse(node) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return "..."


# Unfenced Python detector — finds blocks of plain Python source that are
# not inside markdown fences. The common case is someone pasting raw
# `def f(x): ...` content into a chat. We split on double newlines, try
# to parse each chunk, and compress when (a) chunk >= _MIN_BLOCK_CHARS,
# (b) ast.parse succeeds, (c) saved bytes >= 100. We skip ranges that
# overlap an already-compressed skeleton marker (the fenced pass leaves
# `# AST skeleton (saved ...) ...` markers, which are easy to detect by
# their leading `# AST skeleton` substring).
_UNFENCED_PY_HEADER = "# AST skeleton "


def _apply_unfenced_pass(text: str, replace_block, stats: Dict[str, int]) -> str:
    """Second pass on already-fenced-compressed text to catch unfenced Python.

    The `replace_block` callable is the same one used by the fenced pass; we
    build a synthetic Match object that exposes ``group("body")`` and
    ``group("full")`` matching the fenced regex's named-group contract.
    """
    if _UNFENCED_PY_HEADER not in text:
        # No skeleton markers present → fenced pass did nothing. Still
        # try the unfenced pass on the original text.
        pass

    # Find candidate regions: contiguous text that's NOT a fence opener
    # (```python, ```py, ```) and NOT a skeleton marker.
    out: list[str] = []
    cursor = 0
    n = len(text)
    while cursor < n:
        # Skip forward past any skeleton marker (5+ lines starting with # AST skeleton)
        if text[cursor:cursor + len(_UNFENCED_PY_HEADER)] == _UNFENCED_PY_HEADER:
            nl = text.find("\n\n", cursor)
            if nl == -1:
                out.append(text[cursor:])
                break
            out.append(text[cursor:nl])
            cursor = nl
            continue
        # Find the next fence or skeleton marker
        next_fence = text.find("\n```", cursor)
        next_skel = text.find(_UNFENCED_PY_HEADER, cursor)
        candidates = [c for c in (next_fence, next_skel) if c != -1]
        if not candidates:
            seg_end = n
        else:
            seg_end = min(candidates)
        # Ensure forward progress even on empty segments (e.g. when cursor
        # sits exactly on a fence newline or skeleton marker start).
        if seg_end <= cursor:
            seg_end = cursor + 1
        segment = text[cursor:seg_end]
        # Try the unfenced compression on this segment.
        out.append(_compress_unfenced_segment(segment, replace_block, stats))
        cursor = seg_end

    return "".join(out)


def _compress_unfenced_segment(segment: str, replace_block, stats: Dict[str, int]) -> str:
    """Find unfenced Python chunks inside `segment` and compress them.

    Splits on double-newlines (paragraph boundaries). Each chunk that
    (a) is >= _MIN_BLOCK_CHARS long, (b) parses as Python via ast.parse,
    (c) saves at least 100 bytes after skeletonization, gets replaced
    via the same `replace_block` path the fenced pass uses.
    """
    if len(segment) < _MIN_BLOCK_CHARS:
        return segment

    # Split on double newlines; preserve the separators so we can reassemble.
    parts = re.split(r"(\n\n+)", segment)
    out: list[str] = []
    for part in parts:
        # Skip separators (whitespace) and tiny parts.
        if not part or part.startswith("\n") or len(part) < _MIN_BLOCK_CHARS:
            out.append(part)
            continue
        # Try to parse as Python.
        try:
            ast.parse(part)
        except SyntaxError:
            out.append(part)
            continue
        # Valid Python! Compress via the shared replace_block.
        # Build a synthetic Match that the fenced replace_block can consume.
        class _M:
            def __init__(self, body):
                self._body = body
            def group(self, name):
                if name == "body":
                    return self._body
                if name == "full":
                    return self._body
                return self._body
        stats["blocks_found"] += 1
        # Reuse the fenced replace_block to keep behavior consistent.
        new = replace_block(_M(part))
        # replace_block returns the skeleton marker on success, or `part`
        # unchanged on skip (too small, no savings, parse fail). Only the
        # marker path increments blocks_compressed/bytes_saved inside the
        # closure.
        out.append(new)
    return "".join(out)
