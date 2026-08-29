"""Shared alias-registry loader.

Implements the allowlisted-fragment gate from the canonical Hermes
aliases library design (docs/architecture/hermes-aliases-design.md §3.1,
§4, §6.1). The registry is a single file ``hermes_aliases.yaml`` at the
lane-home root, declaring ``version`` / ``schema`` / ``aliases`` (and
optional ``metadata``). A profile opts in by adding::

    imports:
      - hermes_aliases.yaml

to its config.yaml; the loader then merges the registry's aliases into
the same precedence chain that ``_load_direct_aliases`` already uses
(see ``hermes_cli/model_switch.py``).

Migration-safety contract (design §6.1, §6.4):

- A profile that does NOT declare ``imports:`` MUST observe no
  behavior change. The loader never opens the registry file unless
  ``_should_import_alias_registry(cfg)`` returns True.
- Missing file, unparseable file, wrong top-level shape, per-entry
  type violations, all degrade soft: the loader logs a warning and
  returns an empty contribution. The profile's local ``model.aliases:``
  block is unaffected.
- ``HERMES_ALIAS_REGISTRY_STRICT=1`` opts into strict mode: any
  registry defect raises instead of degrading. Off by default.

The registry parser is deliberately permissive (tolerant YAML,
non-strict key/value shapes) because the design keeps the file's
shape aligned with the existing ``model.aliases`` string-format
contract. Dict-format ``base_url`` overrides are NOT supported here
— profiles that need a custom ``base_url`` use the higher-precedence
``model_aliases:`` surface, which the existing loader already handles.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

from hermes_cli.model_switch import DirectAlias

log = logging.getLogger(__name__)

# Single allowlisted fragment name (design §2.1, §3.1, §4).
ALIAS_REGISTRY_FILENAME = "hermes_aliases.yaml"

# Per-entry validation regexes (design §6.6).
_ALIAS_KEY_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_PROVIDER_MODEL_RE = re.compile(r"^[a-z0-9_.-]+/[a-z0-9_.-]+$")


def _resolve_registry_paths() -> list[Path]:
    """Return candidate paths to the registry file, primary first.

    See ``_get_lane_home_root`` for layout rules. Used by
    ``_load_alias_registry_from_disk`` to handle the per-profile vs.
    lane-root layout ambiguity without an environment probe.
    """
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    candidates: list[Path] = [home / ALIAS_REGISTRY_FILENAME]
    if home.parent.name == "profiles":
        candidates.append(home.parent.parent / ALIAS_REGISTRY_FILENAME)
    return candidates


def _should_import_alias_registry(cfg: Any) -> bool:
    """Allowlisted-fragment gate (design §3.1, §4, §6.1).

    Returns True only when the profile's config explicitly lists
    ``hermes_aliases.yaml`` in its ``imports:`` block. Any other shape
    (missing key, wrong type, different fragment name) returns False,
    which means the registry file is never opened and the profile's
    local ``model.aliases:`` block is used as-is.
    """
    if not isinstance(cfg, dict):
        return False
    imports = cfg.get("imports")
    if not isinstance(imports, (list, tuple)):
        return False
    return any(
        isinstance(item, str) and item.strip() == ALIAS_REGISTRY_FILENAME
        for item in imports
    )


def _strict_mode_enabled() -> bool:
    """Opt-in strict mode env var (design §4.2). Off by default."""
    return os.environ.get("HERMES_ALIAS_REGISTRY_STRICT", "").strip() == "1"


def _parse_alias_value(value: Any) -> Optional[DirectAlias]:
    """Parse one registry entry into a ``DirectAlias``.

    Accepts the canonical string format ``provider/model``. Anything
    else is dropped (with a debug log line) — dict-shaped entries with
    explicit ``base_url`` are NOT supported in the shared file; profiles
    that need them use the higher-precedence ``model_aliases:`` surface.
    """
    if not isinstance(value, str):
        log.debug("alias_registry: dropping non-string entry %r", value)
        return None
    val = value.strip()
    if not val:
        log.debug("alias_registry: dropping empty string entry")
        return None
    if "/" not in val:
        log.debug(
            "alias_registry: dropping malformed entry %r (expected provider/model)",
            val,
        )
        return None
    provider, model = val.split("/", 1)
    provider = provider.strip()
    model = model.strip()
    if not provider or not model or not _PROVIDER_MODEL_RE.match(val):
        log.debug(
            "alias_registry: dropping entry %r (failed provider/model regex)",
            val,
        )
        return None
    return DirectAlias(model=model, provider=provider, base_url="")


def _load_alias_registry_from_disk() -> Dict[str, DirectAlias]:
    """Read ``hermes_aliases.yaml`` from disk and return a parsed alias table.

    Soft-degrades on every failure mode enumerated in design §4.2 /
    §6.4 unless ``HERMES_ALIAS_REGISTRY_STRICT=1`` is set:

    - No file at any candidate path → empty dict (no warning; this is
      the steady state for profiles that never opt in, and for fresh
      lanes that don't yet have a registry).
    - File present but unparseable → warning + empty dict.
    - Wrong top-level shape (not a mapping) → warning + empty dict.
    - Per-entry type violations → drop the bad entry, keep the rest.
    """
    strict = _strict_mode_enabled()
    result: Dict[str, DirectAlias] = {}

    # Find the registry file across the candidate paths.
    registry_path: Optional[Path] = None
    for candidate in _resolve_registry_paths():
        if candidate.is_file():
            registry_path = candidate
            break
    if registry_path is None:
        # No file present. The most common case (profiles that haven't
        # opted in). Silent — no log line. The gate in
        # ``_should_import_alias_registry`` is what blocks the read in
        # the first place; this is the second layer.
        return result

    # Parse the YAML.
    try:
        from utils import fast_safe_load
    except Exception:  # pragma: no cover — defensive
        if strict:
            raise
        log.warning(
            "alias_registry: utils.fast_safe_load unavailable; skipping registry load"
        )
        return result
    try:
        with open(registry_path, encoding="utf-8") as f:
            data = fast_safe_load(f)
    except Exception as e:
        msg = f"alias_registry: failed to parse {registry_path}: {e}"
        if strict:
            raise RuntimeError(msg) from e
        log.warning(msg)
        return result

    # Top-level shape check.
    if not isinstance(data, dict):
        msg = f"alias_registry: {registry_path} top-level is not a mapping"
        if strict:
            raise RuntimeError(msg)
        log.warning(msg)
        return result

    aliases = data.get("aliases")
    if not isinstance(aliases, dict):
        # A profile can opt in before the registry has been authored;
        # an absent ``aliases:`` block is the legitimate "empty
        # contribution" case.
        return result

    for name, raw_value in aliases.items():
        if not isinstance(name, str) or not _ALIAS_KEY_RE.match(name):
            log.debug("alias_registry: dropping key %r (failed key regex)", name)
            continue
        direct = _parse_alias_value(raw_value)
        if direct is None:
            continue
        result[name.strip().lower()] = direct
    return result


def load_alias_registry(cfg: Any) -> Dict[str, DirectAlias]:
    """Public entry point: merge alias registry into the alias table.

    Returns an empty dict unless ``_should_import_alias_registry(cfg)``
    is True. The merged shape is the same as ``_load_direct_aliases``'s
    ``merged`` dict, so the caller can update its own merged dict with
    the return value (precedence: registry entries lose to any key
    already present in the caller's dict, mirroring the existing
    ``model.aliases`` override path).
    """
    if not _should_import_alias_registry(cfg):
        return {}
    return _load_alias_registry_from_disk()
