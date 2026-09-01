"""Behavioral regressions for the v39 saved-toolset cleanup."""

from __future__ import annotations

import os
from unittest.mock import patch

import yaml

from hermes_cli.config_migrations import _migrate_to_39


def _write_config(tmp_path, *, platform_toolsets, known_toolsets):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "platform_toolsets": {"cli": platform_toolsets},
                "known_builtin_toolsets": {"cli": known_toolsets},
                "stt": {"enabled": True, "provider": "local"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return config_path


def _run_v39_migration(tmp_path):
    results = {"env_added": [], "config_added": [], "warnings": []}
    with patch.dict(os.environ, {"HERMES_HOME": str(tmp_path)}):
        _migrate_to_39(results, quiet=True)


def test_v39_removes_stale_saved_stt_but_preserves_dynamic_toolsets(tmp_path):
    config_path = _write_config(
        tmp_path,
        platform_toolsets=["hermes-cli", "a2a", "fleet_bus", "stt", "bfl"],
        known_toolsets=["a2a", "fleet_bus", "stt", "bfl"],
    )

    _run_v39_migration(tmp_path)

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert raw["platform_toolsets"]["cli"] == ["hermes-cli", "a2a", "fleet_bus"]
    assert raw["known_builtin_toolsets"]["cli"] == ["a2a", "fleet_bus"]
    assert raw["stt"] == {"enabled": True, "provider": "local"}


def test_v39_saved_toolset_cleanup_is_byte_idempotent(tmp_path):
    config_path = _write_config(
        tmp_path,
        platform_toolsets=["a2a", "stt", "stt"],
        known_toolsets=["fleet_bus", "stt", "stt"],
    )

    _run_v39_migration(tmp_path)
    first_pass = config_path.read_bytes()
    _run_v39_migration(tmp_path)
    second_pass = config_path.read_bytes()

    raw = yaml.safe_load(second_pass)
    assert "stt" not in raw["platform_toolsets"]["cli"]
    assert "stt" not in raw["known_builtin_toolsets"]["cli"]
    assert second_pass == first_pass
