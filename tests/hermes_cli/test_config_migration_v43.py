"""Migration collision bridge: both v40 histories and versionless configs."""

import copy

import pytest
import yaml

from hermes_cli.config import DEFAULT_CONFIG, migrate_config
from hermes_cli.config_migrations import _migrate_to_43


@pytest.mark.parametrize(
    ("version", "catalog", "expected_minutes"),
    [(40, {"ttl_hours": 1}, 60),
     (40, {"ttl_minutes": 25}, 25),
     (None, {"ttl_hours": 2.5}, 150)],
    ids=["custom-v40", "upstream-v40", "versionless"],
)
def test_bridge_preserves_provider_configuration(tmp_path, monkeypatch, version, catalog, expected_minutes):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    # This test exercises config migration, not the independent SOUL roster cleanup.
    monkeypatch.setattr("tools.bot_mode_probe._roster", lambda root: [])
    protected = {
        "model": {"default": "fixture-model", "provider": "custom:fixture-route"},
        "providers": {"fixture-route": {"api": "openai-completions", "base_url": "http://127.0.0.1:1/v1"}},
        "memory": {"provider": "mnemosyne", "custom_receipt": "fixture-only"},
        "stt": {"enabled": True, "provider": "local"},
    }
    config = {
        **copy.deepcopy(protected),
        "model_catalog": catalog,
        "platform_toolsets": {"cli": ["a2a", "fleet_bus", "stt", "bfl"]},
        "known_builtin_toolsets": {"cli": ["fleet_bus", "stt", "bfl"]},
    }
    if version is not None:
        config["_config_version"] = version
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    migrate_config(interactive=False, quiet=True)

    actual = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert actual["_config_version"] == DEFAULT_CONFIG["_config_version"]
    assert actual["model_catalog"]["ttl_minutes"] == expected_minutes
    assert "ttl_hours" not in actual["model_catalog"]
    for key, expected in protected.items():
        assert actual[key] == expected
    assert actual["platform_toolsets"]["cli"] == ["a2a", "fleet_bus"]
    assert actual["known_builtin_toolsets"]["cli"] == ["fleet_bus"]
    first = path.read_bytes()
    migrate_config(interactive=False, quiet=True)
    assert path.read_bytes() == first


@pytest.mark.parametrize("catalog", [{"ttl_hours": True}, {"ttl_hours": "bad"}, {"ttl_hours": 1, "ttl_minutes": 25}])
def test_bridge_does_not_guess_malformed_or_override_minutes(tmp_path, monkeypatch, catalog):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"_config_version": 40, "model_catalog": catalog}), encoding="utf-8")
    before = path.read_bytes()
    _migrate_to_43({"config_added": []}, quiet=True)
    assert path.read_bytes() == before
