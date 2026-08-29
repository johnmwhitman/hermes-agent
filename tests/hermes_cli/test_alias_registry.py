"""Shared alias-registry tests.

Covers the allowlisted-fragment gate introduced by the canonical Hermes
aliases library design (docs/architecture/hermes-aliases-design.md §3.1,
§4, §5, §6.1, §6.4). Tests run in hermetic isolation: each test sets
HERMES_HOME to a tmp_path, writes a hermes_aliases.yaml there, and
exercises the gate / parser / merger independently. The resolution-
consistency assertion is the only test that compares two profile
configurations against each other (it stands in for the cross-profile
invariant the design calls out as the primary correctness guarantee).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from hermes_cli.alias_registry import (
    ALIAS_REGISTRY_FILENAME,
    _parse_alias_value,
    _resolve_registry_paths,
    _should_import_alias_registry,
    load_alias_registry,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def hermetic_lane_home(tmp_path, monkeypatch):
    """Point HERMES_HOME at a fresh tmp_path lane root.

    The lane root is ``tmp_path``; profiles would live under
    ``tmp_path/profiles/<name>/`` if a test needed that. The registry
    file goes directly at ``tmp_path/hermes_aliases.yaml`` because
    lane-root lookups are the most common deployment shape.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # The hermes-agent test suite has its own conftest that mucks with
    # HERMES_HOME via context-local overrides; assert we own the env.
    assert os.environ["HERMES_HOME"] == str(tmp_path)
    return tmp_path


@pytest.fixture
def hermetic_profile_home(tmp_path, monkeypatch):
    """Point HERMES_HOME at a fresh ``tmp_path/profiles/<name>/`` directory.

    Exercises the per-profile launch shape: registry lookup walks up
    to ``tmp_path/`` (the lane root) where ``hermes_aliases.yaml``
    lives. Mirrors the production routeplane/conductor shape.
    """
    profile_dir = tmp_path / "profiles" / "test-profile"
    profile_dir.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(profile_dir))
    return tmp_path, profile_dir


def _write_registry(lane_root: Path, body: str) -> Path:
    p = lane_root / ALIAS_REGISTRY_FILENAME
    p.write_text(body, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Gate tests (design §3.1, §4, §6.1) — behavior change iff imports: declared
# ---------------------------------------------------------------------------


def test_gate_no_imports_key_returns_false():
    """Profiles without ``imports:`` never observe the registry (design §6.1)."""
    assert _should_import_alias_registry({}) is False
    assert _should_import_alias_registry({"model": {"aliases": {"minimax": "ollama/x"}}}) is False
    assert _should_import_alias_registry({"imports": []}) is False


def test_gate_with_correct_filename_returns_true():
    """The allowlisted filename is the only thing that flips the gate."""
    cfg = {"imports": [ALIAS_REGISTRY_FILENAME]}
    assert _should_import_alias_registry(cfg) is True


def test_gate_ignores_other_filenames():
    """Whitelist enforcement: unrelated fragments never open the registry."""
    cfg = {"imports": ["some_other.yaml", "random.txt"]}
    assert _should_import_alias_registry(cfg) is False


def test_gate_accepts_filename_among_others():
    """The allowlisted fragment may coexist with unrelated fragments."""
    cfg = {"imports": ["plugins/foo.yaml", ALIAS_REGISTRY_FILENAME]}
    assert _should_import_alias_registry(cfg) is True


def test_gate_non_dict_config_returns_false():
    """Defensive: a non-dict cfg (legacy shape, parse error) must not crash."""
    assert _should_import_alias_registry(None) is False
    assert _should_import_alias_registry("imports: [x]") is False
    assert _should_import_alias_registry(["x"]) is False


def test_gate_imports_must_be_list_not_string():
    """A bare string ``imports: 'foo.yaml'`` is not a list — gate stays off."""
    cfg = {"imports": ALIAS_REGISTRY_FILENAME}
    assert _should_import_alias_registry(cfg) is False


def test_gate_filename_whitespace_normalized():
    """``'  hermes_aliases.yaml  '`` is still the allowlisted fragment."""
    cfg = {"imports": ["  " + ALIAS_REGISTRY_FILENAME + "  "]}
    assert _should_import_alias_registry(cfg) is True


# ---------------------------------------------------------------------------
# Path resolution (per-profile vs. lane-root layout)
# ---------------------------------------------------------------------------


def test_resolve_paths_lane_root_layout(hermetic_lane_home):
    """HERMES_HOME IS the lane root → primary candidate is the root itself."""
    cands = _resolve_registry_paths()
    assert cands[0] == hermetic_lane_home / ALIAS_REGISTRY_FILENAME
    assert len(cands) == 1


def test_resolve_paths_per_profile_layout(hermetic_profile_home):
    """HERMES_HOME is ``<root>/profiles/<name>/`` → primary is the profile,
    fallback is the lane root."""
    lane_root, profile_dir = hermetic_profile_home
    cands = _resolve_registry_paths()
    assert cands[0] == profile_dir / ALIAS_REGISTRY_FILENAME
    assert lane_root / ALIAS_REGISTRY_FILENAME in cands


# ---------------------------------------------------------------------------
# Per-entry validation
# ---------------------------------------------------------------------------


def test_parse_alias_value_canonical():
    """Canonical ``provider/model`` shape parses."""
    direct = _parse_alias_value("ollama/qwen3.5")
    assert direct is not None
    assert direct.model == "qwen3.5"
    assert direct.provider == "ollama"
    assert direct.base_url == ""


def test_parse_alias_value_drops_malformed():
    """Entries missing a ``/`` separator are dropped (base_url override
    belongs in model_aliases:, not the registry)."""
    assert _parse_alias_value("ollama") is None
    assert _parse_alias_value("") is None
    assert _parse_alias_value("  ") is None
    assert _parse_alias_value(123) is None
    assert _parse_alias_value(None) is None
    assert _parse_alias_value({"provider": "x", "model": "y"}) is None


def test_parse_alias_value_drops_bad_chars():
    """Entries with uppercase or spaces are rejected by the regex."""
    assert _parse_alias_value("Ollama/Qwen") is None
    assert _parse_alias_value("ollama / qwen") is None


def test_parse_alias_value_drops_empty_provider_or_model():
    """``/model`` or ``provider/`` with one side empty is rejected."""
    assert _parse_alias_value("/qwen") is None
    assert _parse_alias_value("ollama/") is None


# ---------------------------------------------------------------------------
# Disk loader — soft-degradation paths
# ---------------------------------------------------------------------------


def test_load_missing_file_returns_empty(hermetic_lane_home):
    """No file → empty dict, no exception (steady state for non-opt-in)."""
    cfg = {"imports": [ALIAS_REGISTRY_FILENAME]}
    assert load_alias_registry(cfg) == {}


def test_load_missing_file_no_imports_returns_empty(hermetic_lane_home):
    """No file AND no imports: → empty, never reads disk."""
    # _should_import_alias_registry short-circuits; even if a file were
    # present it would not be opened.
    _write_registry(hermetic_lane_home, "aliases:\n  minimax: ollama/x\n")
    assert load_alias_registry({}) == {}


def test_load_well_formed_registry(hermetic_lane_home):
    """A correct registry yields the parsed DirectAlias table."""
    _write_registry(
        hermetic_lane_home,
        "version: 1\naliases:\n  minimax: subs/minimax\n  qwen: ollama/qwen3.5\n",
    )
    cfg = {"imports": [ALIAS_REGISTRY_FILENAME]}
    out = load_alias_registry(cfg)
    assert set(out) == {"minimax", "qwen"}
    assert out["minimax"].model == "minimax"
    assert out["minimax"].provider == "subs"
    assert out["qwen"].model == "qwen3.5"
    assert out["qwen"].provider == "ollama"


def test_load_per_profile_finds_lane_root(hermetic_profile_home):
    """HERMES_HOME is a profile dir → loader walks up to find the registry."""
    lane_root, _profile = hermetic_profile_home
    _write_registry(lane_root, "aliases:\n  minimax: subs/minimax\n")
    cfg = {"imports": [ALIAS_REGISTRY_FILENAME]}
    out = load_alias_registry(cfg)
    assert "minimax" in out


def test_load_unparseable_yaml_warns_and_returns_empty(hermetic_lane_home, caplog):
    """Broken YAML degrades to empty (soft degradation, design §4.2)."""
    _write_registry(hermetic_lane_home, "this is: not: valid: yaml: ::\n  bad indent")
    cfg = {"imports": [ALIAS_REGISTRY_FILENAME]}
    with caplog.at_level("WARNING"):
        out = load_alias_registry(cfg)
    assert out == {}


def test_load_top_level_not_mapping_warns(hermetic_lane_home, caplog):
    """A list at the top level is not a valid registry."""
    _write_registry(hermetic_lane_home, "- minimax\n- qwen\n")
    cfg = {"imports": [ALIAS_REGISTRY_FILENAME]}
    with caplog.at_level("WARNING"):
        out = load_alias_registry(cfg)
    assert out == {}


def test_load_drops_bad_entries_keeps_good(hermetic_lane_home):
    """Per-entry violations are dropped, valid entries survive."""
    _write_registry(
        hermetic_lane_home,
        (
            "aliases:\n"
            "  minimax: subs/minimax\n"            # good
            "  qwen: ollama/qwen3.5\n"              # good
            "  bad1: 'no-slash-here'\n"             # bad — no slash
            "  bad2: ''\n"                          # bad — empty
            "  bad3: '/only-model'\n"               # bad — empty provider
            "  Bad-Upper: subs/x\n"                 # bad — uppercase key
            "  has space: subs/x\n"                 # bad — space in key
        ),
    )
    cfg = {"imports": [ALIAS_REGISTRY_FILENAME]}
    out = load_alias_registry(cfg)
    assert set(out) == {"minimax", "qwen"}


def test_load_strict_mode_raises_on_bad_yaml(hermetic_lane_home, monkeypatch):
    """``HERMES_ALIAS_REGISTRY_STRICT=1`` opts into raising on defects."""
    monkeypatch.setenv("HERMES_ALIAS_REGISTRY_STRICT", "1")
    _write_registry(hermetic_lane_home, "this is: not: valid: yaml:\n  bad")
    cfg = {"imports": [ALIAS_REGISTRY_FILENAME]}
    with pytest.raises(RuntimeError, match="alias_registry"):
        load_alias_registry(cfg)


def test_load_strict_mode_off_by_default(hermetic_lane_home):
    """Without the env var, broken YAML silently degrades (default behavior)."""
    assert os.environ.get("HERMES_ALIAS_REGISTRY_STRICT", "") != "1"
    _write_registry(hermetic_lane_home, "this is: not: valid: yaml:\n  bad")
    cfg = {"imports": [ALIAS_REGISTRY_FILENAME]}
    # Must not raise.
    assert load_alias_registry(cfg) == {}


# ---------------------------------------------------------------------------
# Precedence — registry loses to profile-local override (design §3.1)
# ---------------------------------------------------------------------------


def test_profile_local_model_alias_dict_wins_over_registry(hermetic_lane_home, monkeypatch):
    """The dict-format ``model_aliases:`` surface wins over registry.

    This is the conductor pattern: dict entries land first in
    ``_load_direct_aliases``'s ``merged`` dict, and registry entries
    are guarded by ``if name in merged: continue``.
    """
    from hermes_cli import model_switch

    _write_registry(hermetic_lane_home, "aliases:\n  minimax: ollama/glm-5.2\n")
    monkeypatch.setattr(
        model_switch, "_BUILTIN_DIRECT_ALIASES", {}, raising=False,
    )

    from hermes_cli.config import load_config

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "imports": [ALIAS_REGISTRY_FILENAME],
            "model": {"provider": "x", "aliases": {}},
            "model_aliases": {"minimax": {"model": "y", "provider": "z", "base_url": ""}},
        },
    )
    out = model_switch._load_direct_aliases()
    # Profile-local override won: ollama/glm-5.2 from registry is NOT here.
    assert out["minimax"].provider == "z"
    assert out["minimax"].model == "y"


def test_profile_local_model_alias_string_wins_over_registry(hermetic_lane_home, monkeypatch):
    """The string-format ``model.aliases:`` block wins over registry too.

    Mirrors the conductor, meshfleet, designer, fleetopus migration
    pattern: each profile's local ``minimax: subs/minimax`` (or
    equivalent) is preserved when it overrides the registry.
    """
    from hermes_cli import model_switch

    _write_registry(hermetic_lane_home, "aliases:\n  minimax: ollama/glm-5.2\n")
    monkeypatch.setattr(
        model_switch, "_BUILTIN_DIRECT_ALIASES", {}, raising=False,
    )

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "imports": [ALIAS_REGISTRY_FILENAME],
            "model": {
                "provider": "custom",
                "aliases": {"minimax": "subs/minimax"},
            },
            "model_aliases": {},
        },
    )
    out = model_switch._load_direct_aliases()
    assert out["minimax"].provider == "subs"
    assert out["minimax"].model == "minimax"


def test_registry_supplies_aliases_only_when_profile_does_not(hermetic_lane_home, monkeypatch):
    """When the profile has no local entry, the registry supplies it."""
    from hermes_cli import model_switch

    _write_registry(
        hermetic_lane_home,
        "aliases:\n  minimax: subs/minimax\n  qwen: ollama/qwen3.5\n",
    )
    monkeypatch.setattr(
        model_switch, "_BUILTIN_DIRECT_ALIASES", {}, raising=False,
    )

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "imports": [ALIAS_REGISTRY_FILENAME],
            "model": {"provider": "custom", "aliases": {}},
            "model_aliases": {},
        },
    )
    out = model_switch._load_direct_aliases()
    assert out["minimax"].model == "minimax"
    assert out["qwen"].model == "qwen3.5"


# ---------------------------------------------------------------------------
# Resolution-consistency invariant (design §5.1) — the cross-profile check
# ---------------------------------------------------------------------------


def test_resolution_consistency_same_canonical_alias_across_profiles(
    hermetic_lane_home, monkeypatch,
):
    """The canonical alias ``minimax`` resolves to the same DirectAlias
    regardless of which opt-in profile is asking.

    Design §5.1: this is the primary correctness guarantee of the shared
    library — without it, two profiles can drift and break the A2A hub,
    kanban handoffs, and billing reports that key off the alias.
    """
    from hermes_cli import model_switch

    _write_registry(
        hermetic_lane_home,
        "aliases:\n  minimax: subs/minimax\n  qwen: ollama/qwen3.5\n",
    )
    monkeypatch.setattr(
        model_switch, "_BUILTIN_DIRECT_ALIASES", {}, raising=False,
    )

    # Profile A: minimal config (no local override).
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "imports": [ALIAS_REGISTRY_FILENAME],
            "model": {"provider": "x", "aliases": {}},
            "model_aliases": {},
        },
    )
    out_a = model_switch._load_direct_aliases()

    # Profile B: heavy local config with overlapping entries that DO NOT
    # touch ``minimax`` (proves registry is the single source for keys
    # the profile did not override).
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "imports": [ALIAS_REGISTRY_FILENAME],
            "model": {
                "provider": "custom",
                "aliases": {
                    "grok": "subs/grok",
                    "kimi": "moonshotai/kimi-k3",
                    # note: minimax NOT in here
                },
            },
            "model_aliases": {},
        },
    )
    out_b = model_switch._load_direct_aliases()

    assert out_a["minimax"] == out_b["minimax"]
    # And qwen is the same too — both profiles inherited from the registry.
    assert out_a["qwen"] == out_b["qwen"]


def test_resolution_consistency_preserves_routeplane_only_qwen_trio(hermetic_lane_home, monkeypatch):
    """routeplane keeps its routeplane-only qwen trio (qwencloud,
    qwen38max, qwen37plus) as local overrides; the registry's defaults
    for OTHER keys still flow through.

    This is the design §3.6 / §6.5 carve-out: routeplane has documented
    provider-scoped aliases that fail on other profiles today. They
    stay in the routeplane profile, not the shared registry.
    """
    from hermes_cli import model_switch

    _write_registry(
        hermetic_lane_home,
        "aliases:\n"
        "  minimax: subs/minimax\n"
        "  qwen: ollama/qwen3.5\n"  # registry default
        "  qwencloud: qwencloud/qwen3.7-plus\n"  # same as local
    )
    monkeypatch.setattr(
        model_switch, "_BUILTIN_DIRECT_ALIASES", {}, raising=False,
    )

    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "imports": [ALIAS_REGISTRY_FILENAME],
            "model": {
                "provider": "routeplane",
                "aliases": {
                    "qwencloud": "qwencloud/qwen3.7-plus",     # local override
                    "qwen38max": "qwen/qwen3.8-max",           # routeplane-only
                    "qwen37plus": "qwen/qwen3.7-plus",         # routeplane-only
                },
            },
            "model_aliases": {},
        },
    )
    out = model_switch._load_direct_aliases()
    # routeplane-only trio preserved:
    assert out["qwencloud"].provider == "qwencloud"
    assert out["qwen38max"].provider == "qwen"
    assert out["qwen37max" if False else "qwen37plus"].provider == "qwen"
    # canonical keys inherited from registry:
    assert out["minimax"].provider == "subs"
    assert out["qwen"].provider == "ollama"


# ---------------------------------------------------------------------------
# Migration-safety — design §6.1, §6.4
# ---------------------------------------------------------------------------


def test_migration_no_baseline_profile_gains_full_registry(hermetic_lane_home, monkeypatch):
    """fleetopus had no ``model.aliases:`` block before. With ``imports:``
    declared, it now has the full 29-entry table.

    This is the fleetopus migration (§6.5): the KeyError baseline from
    the parent audit (t_5a11b49e) is fixed because the registry
    supplies every alias the profile never declared.
    """
    from hermes_cli import model_switch

    _write_registry(
        hermetic_lane_home,
        (
            "aliases:\n"
            "  minimax: subs/minimax\n"
            "  grok: subs/grok\n"
            "  codex: subs/codex\n"
            "  kimi: moonshotai/kimi-k3\n"
            "  qwen: ollama/qwen3.5\n"
        ),
    )
    monkeypatch.setattr(
        model_switch, "_BUILTIN_DIRECT_ALIASES", {}, raising=False,
    )

    # fleetopus-shaped config: no model.aliases block, no model_aliases.
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "imports": [ALIAS_REGISTRY_FILENAME],
            "model": {"provider": "routeplane", "default": "routeplane/auto"},
            "model_aliases": {},
        },
    )
    out = model_switch._load_direct_aliases()
    # All 5 declared aliases are now resolvable — no more KeyError.
    assert all(k in out for k in ("minimax", "grok", "codex", "kimi", "qwen"))


def test_no_imports_means_zero_behavior_change(hermetic_lane_home, monkeypatch):
    """A profile with no ``imports:`` block gets an empty contribution,
    even when a registry file exists on disk.

    Design §6.1 contract: this is the migration-safety guarantee.
    """
    from hermes_cli import model_switch

    _write_registry(
        hermetic_lane_home,
        "aliases:\n  minimax: subs/minimax\n",
    )
    monkeypatch.setattr(
        model_switch, "_BUILTIN_DIRECT_ALIASES", {}, raising=False,
    )

    # No imports: key — registry must NOT contribute.
    monkeypatch.setattr(
        "hermes_cli.config.load_config",
        lambda: {
            "model": {"provider": "x", "aliases": {}},
            "model_aliases": {},
        },
    )
    out = model_switch._load_direct_aliases()
    assert "minimax" not in out
