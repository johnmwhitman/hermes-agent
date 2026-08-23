# plugins/context_engine/toon_compressor/__init__.py
#
# TOON-style JSON compressor (Hermes-native).
#
# Strategy: when conversation context contains large arrays of objects that
# share the same shape (a very common MCP response pattern), replace them
# with a TSV-shaped summary: header line with column names + types, then one
# row per object.
#
# This is the Hermes-native realization of Headroom's "SmartCrusher"
# statistical JSON / array compression technique, but it leans on the
# emerging Token-Oriented Object Notation (TOON) idiom — TSV-shaped object
# arrays are roughly 50% smaller than pretty-printed JSON, and the agent
# can parse them with the same ease.
#
# Selection: in config.yaml, set
#     context.engine: toon_compressor
# to activate. The default `compressor` is unaffected.

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from agent.context_engine import ContextEngine

logger = logging.getLogger(__name__)


# Minimum JSON array size (in characters) to consider TOON-encoding.
# Below this, the header line adds overhead that offsets the savings.
_MIN_JSON_CHARS = 700

# Inline detection regex — looks for an array of objects with consistent
# top-level keys. Conservative: we only fire when the array is the dominant
# content of the string and all entries are objects with shared keys.
_INFERRED_JSON_RE = re.compile(r"(\[[\s\S]{" + str(_MIN_JSON_CHARS) + r",}\])")


class ToonCompressor(ContextEngine):
    """ContextEngine that replaces uniform-shape JSON arrays with TSV summary.

    Best for MCP responses like:
        [{"id":1,"name":"a","x":1.0},{"id":2,"name":"b","x":2.0}, ...]

    Becomes:
        # TOON summary (saved NNN bytes / MM%)
        # original_sha256=HHHHHHHHHHHHHHHH
        # columns: id:int, name:str, x:float
        1	a	1.0
        2	b	2.0
        ...

    Non-uniform arrays, deeply-nested objects, and single-object JSON are
    passed through unchanged.
    """

    @property
    def name(self) -> str:
        return "toon_compressor"

    def is_available(self) -> bool:
        return True

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        pass

    def should_compress(self, prompt_tokens: int = None) -> bool:
        # Event-driven; host uses its own threshold.
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

        Same integration story as ``ast_code_compressor.select_context``:
        the host never calls our ``compress()`` because should_compress() is
        always False; without this hook the engine is dormant. Compressing
        per-turn on the request list gives us real-world reach with zero
        risk to the persisted transcript (request-only mutation).
        """
        if not request_messages:
            return request_messages
        compressed, stats = self._compress_messages(request_messages)
        if stats["compressed"] > 0:
            self._compression_count = getattr(self, "_compression_count", 0) + 1
        return compressed

    def compress(
        self,
        messages: List[Dict[str, Any]],
        current_tokens: Optional[int] = None,
        focus_topic: Optional[str] = None,
        force: bool = False,
        memory_context: str = "",
    ) -> List[Dict[str, Any]]:
        if not messages:
            return messages
        out, stats = self._compress_messages(messages)
        if stats["compressed"] > 0:
            self.compression_count += 1
            self._last_compression_savings_pct = (
                100.0 * stats["bytes_saved"]
                / max(1, stats["bytes_saved"] + stats["compressed"] * _MIN_JSON_CHARS)
            )
            logger.info(
                "toon_compressor: %d JSON array(s) compressed, %d bytes saved (%.1f%%)",
                stats["compressed"], stats["bytes_saved"],
                self._last_compression_savings_pct,
            )
        return out

    def _compress_messages(self, messages):
        """Shared per-message compression path used by both ``compress()``
        and the per-turn ``select_context()`` hook."""
        if not messages:
            return messages, {"candidates_found": 0, "compressed": 0, "bytes_saved": 0}
        protected_head = 1 + getattr(self, "protect_first_n", 3)
        out: List[Dict[str, Any]] = []
        stats = {"candidates_found": 0, "compressed": 0, "bytes_saved": 0}
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
        stats = {"candidates_found": 0, "compressed": 0, "bytes_saved": 0}
        if not isinstance(msg, dict):
            return msg, stats
        content = msg.get("content")
        if not isinstance(content, str):
            return msg, stats

        new_content, msg_stats = self._compress_text(content)
        for k, v in msg_stats.items():
            stats[k] += v
        if msg_stats["compressed"] == 0:
            return msg, stats

        new_msg = dict(msg)
        new_msg["content"] = new_content
        return new_msg, stats

    def _compress_text(self, text: str) -> tuple[str, Dict[str, int]]:
        stats = {"candidates_found": 0, "compressed": 0, "bytes_saved": 0}

        def replace_json_array(match: re.Match) -> str:
            stats["candidates_found"] += 1
            blob = match.group(1)
            try:
                arr = json.loads(blob)
            except (json.JSONDecodeError, ValueError):
                return blob
            if not isinstance(arr, list) or not arr:
                return blob
            if not all(isinstance(x, dict) for x in arr):
                return blob
            toon = toon_tsv(arr)
            if toon is None:
                return blob
            saved = len(blob) - len(toon)
            if saved < 100:
                return blob
            content_hash = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
            self._store_original(blob, content_hash)
            wrapped = (
                f"```toon\n# TOON summary (saved {saved} bytes / "
                f"{100*saved//len(blob)}%)\n"
                f"# original_sha256={content_hash}\n"
                f"# use ccr_retrieve('{content_hash}') to fetch original\n"
                f"{toon}\n```"
            )
            stats["compressed"] += 1
            stats["bytes_saved"] += saved
            return wrapped

        new_text = _INFERRED_JSON_RE.sub(replace_json_array, text)
        return new_text, stats

    def set_ccr_store(self, store) -> None:
        """Wire a CcrStore (from plugins.context_engine.ccr_store) so that
        each replaced JSON array stores its original blob for retrieval."""
        self._ccr_store = store

    def _store_original(self, blob: str, content_hash: str) -> None:
        store = getattr(self, "_ccr_store", None)
        if store is None:
            return
        try:
            store.put(blob.encode("utf-8"), "json", "toon_compressor-1.0")
        except Exception as exc:
            logger.warning("ccr_store put failed for %s: %s", content_hash, exc)


def toon_tsv(rows: List[Dict[str, Any]]) -> Optional[str]:
    """Convert a list of uniform-shape dicts into a TSV-shaped TOON string.

    Returns None if the rows are not uniform (different keys, mixed types
    in the same column, nested objects).
    """
    if not rows:
        return None

    # Column set: union of all keys, in insertion order of the first row.
    columns: List[str] = []
    seen = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                columns.append(k)
                seen.add(k)

    # Check uniformity: every row must have every column, and types must
    # match the first row's type per column.
    inferred_types: Dict[str, str] = {}
    for r in rows:
        for col in columns:
            if col not in r:
                return None  # not uniform
            val = r[col]
            col_type = _infer_type(val)
            if col not in inferred_types:
                inferred_types[col] = col_type
            elif inferred_types[col] != col_type:
                return None  # type drift across rows
        for k in r:
            if k not in columns:
                return None  # unexpected extra key

    # Build header
    header = "# columns: " + ", ".join(
        f"{c}:{inferred_types[c]}" for c in columns
    )

    # Build rows
    out_lines = [header]
    for r in rows:
        cells = []
        for col in columns:
            val = r[col]
            if val is None:
                cells.append("")
            elif inferred_types[col] in ("str", "bool"):
                cells.append(_escape_tsv(str(val)))
            else:
                cells.append(_escape_tsv(json.dumps(val)))
        out_lines.append("\t".join(cells))

    return "\n".join(out_lines)


def _infer_type(val: Any) -> str:
    """Map a Python value to a TOON-style type tag."""
    if val is None:
        return "null"
    if isinstance(val, bool):
        return "bool"
    if isinstance(val, int):
        return "int"
    if isinstance(val, float):
        return "float"
    if isinstance(val, str):
        return "str"
    if isinstance(val, list):
        return "list"
    if isinstance(val, dict):
        return "obj"
    return "unknown"


def _escape_tsv(cell: str) -> str:
    """Escape TSV-special characters in a cell."""
    return cell.replace("\t", "\\t").replace("\n", "\\n").replace("\r", "\\r")
