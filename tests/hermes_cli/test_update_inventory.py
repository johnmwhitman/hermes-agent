"""Tests for hermes_cli.update_inventory — the plan phase (#91277 Phase 2)."""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import hermes_cli.update_inventory as ui


def _write_state(home: Path, pid: int, sha: str | None = None, version: str | None = None):
    record = {"pid": pid}
    if sha:
        record["code_sha"] = sha
    if version:
        record["code_version"] = version
    (home / "gateway_state.json").write_text(json.dumps(record), encoding="utf-8")


@pytest.fixture()
def fleet(monkeypatch, tmp_path):
    """Two profiles with live gateways: default (systemd) + work (manual)."""
    default_home = tmp_path / "home"
    work_home = tmp_path / "home" / "profiles" / "work"
    work_home.mkdir(parents=True)
    _write_state(default_home, 100, sha="a" * 40, version="1.0")
    _write_state(work_home, 200)  # pre-stamp gateway: no code identity

    import re
    monkeypatch.setattr("hermes_cli.profiles._get_default_hermes_home", lambda: default_home)
    monkeypatch.setattr("hermes_cli.profiles._get_profiles_root", lambda: default_home / "profiles")
    monkeypatch.setattr("hermes_cli.profiles._PROFILE_ID_RE", re.compile(r"^[a-z0-9][a-z0-9_-]*$"), raising=False)
    monkeypatch.setattr("gateway.status._pid_exists", lambda pid: pid in (100, 200))
    monkeypatch.setattr("hermes_cli.gateway._get_service_pids", lambda all_profiles=False: {100})
    monkeypatch.setattr("hermes_cli.gateway.supports_systemd_services", lambda: True)
    monkeypatch.setattr("hermes_cli.gateway.find_profile_gateway_processes", lambda exclude_pids=None: [])
    monkeypatch.setattr(ui, "_iter_process_cmdlines", lambda: [], raising=False)
    monkeypatch.setattr(
        "hermes_cli.build_info.get_code_identity",
        lambda refresh=False: {"sha": "a" * 40, "short_sha": "a" * 8, "version": "1.0", "source": "git"},
    )
    monkeypatch.setattr("hermes_cli.config.detect_install_method", lambda *a, **k: "git")
    monkeypatch.setattr("hermes_cli.config.get_managed_system", lambda: None)
    return tmp_path


class TestCollectInventory:
    def test_two_profile_fleet(self, fleet):
        plan = ui.collect_runtime_inventory()
        assert plan.install_method == "git"
        assert plan.updatable_in_place is True
        assert plan.expected_sha == "a" * 40
        assert plan.profiles == ["default", "work"]
        assert len(plan.runtimes) == 2
        by_profile = {r.profile: r for r in plan.runtimes}
        assert by_profile["default"].pid == 100
        assert by_profile["default"].supervisor == "systemd"
        assert by_profile["default"].code_sha == "a" * 40
        assert by_profile["work"].pid == 200
        assert by_profile["work"].supervisor == "manual"
        assert by_profile["work"].code_sha is None  # pre-stamp gateway
        assert by_profile["work"].restart_via == "manual"
        from hermes_cli.update_inventory import describe_restart_mechanism

        assert "hermes -p work gateway restart" in describe_restart_mechanism(
            by_profile["work"].restart_via, "work"
        )

    def test_docker_install_not_updatable_in_place(self, fleet, monkeypatch):
        monkeypatch.setattr("hermes_cli.config.detect_install_method", lambda *a, **k: "docker")
        monkeypatch.setattr(
            "hermes_cli.config.recommended_update_command_for_method",
            lambda m: "docker pull nousresearch/hermes-agent:latest",
        )
        plan = ui.collect_runtime_inventory()
        assert plan.install_method == "docker"
        assert plan.updatable_in_place is False
        assert "docker pull" in plan.update_mechanism

    def test_dead_pids_excluded(self, fleet, monkeypatch):
        monkeypatch.setattr("gateway.status._pid_exists", lambda pid: False)
        plan = ui.collect_runtime_inventory()
        assert plan.runtimes == []

    def test_pid_file_fallback_covers_unstamped_profiles(self, fleet, monkeypatch):
        """Gateways with a PID file but no runtime-status record still appear."""
        from hermes_cli.gateway import ProfileGatewayProcess

        monkeypatch.setattr(
            "hermes_cli.gateway.find_profile_gateway_processes",
            lambda exclude_pids=None: [
                ProfileGatewayProcess(profile="legacy", path=Path("/x"), pid=300),
                # duplicate of an already-seen pid — must be deduped
                ProfileGatewayProcess(profile="default", path=Path("/y"), pid=100),
            ],
        )
        monkeypatch.setattr("gateway.status._pid_exists", lambda pid: pid in (100, 200))
        plan = ui.collect_runtime_inventory()
        profiles = [r.profile for r in plan.runtimes]
        assert profiles.count("default") == 1  # deduped by pid
        assert "legacy" in profiles

    def test_never_raises_when_everything_fails(self, monkeypatch):
        def _boom(*a, **k):
            raise RuntimeError("probe down")

        for target in (
            "hermes_cli.config.detect_install_method",
            "hermes_cli.build_info.get_code_identity",
            "hermes_cli.profiles._get_default_hermes_home",
            "hermes_cli.gateway._get_service_pids",
            "hermes_cli.gateway.find_profile_gateway_processes",
        ):
            monkeypatch.setattr(target, _boom)
        monkeypatch.setattr(ui, "_iter_process_cmdlines", _boom)
        plan = ui.collect_runtime_inventory()
        assert plan.runtimes == []
        assert plan.install_method == "unknown"

    def test_plan_serializes_for_receipt(self, fleet):
        plan = ui.collect_runtime_inventory()
        payload = plan.to_dict()
        # must be JSON-clean for the receipt
        text = json.dumps(payload)
        restored = json.loads(text)
        assert restored["install_method"] == "git"
        assert len(restored["runtimes"]) == 2
        assert restored["runtimes"][0]["kind"] == "gateway"

    def test_desktop_serve_worker_is_inventoried_with_code_identity(
        self, fleet, monkeypatch, tmp_path
    ):
        """A Desktop-owned ``serve`` has no gateway_state.json but is live code."""
        runtime_root = tmp_path / "desktop-runtime"
        python = runtime_root / "venv" / "bin" / "python"
        python.parent.mkdir(parents=True)
        (runtime_root / "hermes_cli").mkdir()
        (runtime_root / "pyproject.toml").write_text(
            '[project]\nname = "hermes-agent"\nversion = "9.8.7"\n',
            encoding="utf-8",
        )
        (runtime_root / ".git").mkdir()
        sha = "d" * 40
        (runtime_root / ".git" / "HEAD").write_text(sha + "\n", encoding="utf-8")
        secret_path = "/private/never-report-this-token"
        monkeypatch.setattr(
            ui,
            "_iter_process_cmdlines",
            lambda: [
                (
                    303,
                    [
                        str(python),
                        "-m",
                        "hermes_cli.main",
                        "--profile",
                        "work",
                        "serve",
                        "--host",
                        "127.0.0.1",
                        "--port",
                        "0",
                        "--ssh-session-token-file",
                        secret_path,
                    ],
                )
            ],
            raising=False,
        )

        plan = ui.collect_runtime_inventory()

        serve = next(r for r in plan.runtimes if r.pid == 303)
        assert serve.kind == "serve"
        assert serve.profile == "work"
        assert serve.supervisor == "desktop"
        assert serve.restart_via == "desktop"
        assert serve.code_sha == sha
        assert serve.code_version == "9.8.7"
        serialized = json.dumps(serve.to_dict())
        assert secret_path not in serialized
        assert "ssh-session-token-file" not in serialized

    def test_desktop_serve_scan_is_strict_and_deduplicates_gateway_pid(
        self, fleet, monkeypatch
    ):
        monkeypatch.setattr(
            ui,
            "_iter_process_cmdlines",
            lambda: [
                (100, ["python", "-m", "hermes_cli.main", "serve", "--port", "0"]),
                (301, ["python", "-m", "hermes_cli.main", "dashboard", "--port", "0"]),
                (302, ["python", "-m", "hermes_cli.main", "serve", "--port", "9119"]),
                (303, ["notes", "about", "hermes_cli.main", "serve", "--port", "0"]),
                (304, ["python", "-m", "unrelated.module", "serve", "--port", "0"]),
            ],
            raising=False,
        )

        plan = ui.collect_runtime_inventory()

        assert [r.pid for r in plan.runtimes].count(100) == 1
        assert not ({301, 302, 303, 304} & {r.pid for r in plan.runtimes})


class TestDesktopServeCommandParser:
    def test_accepts_module_entrypoint_and_profile_flag_forms(self):
        assert ui._parse_desktop_serve_command(
            "/runtime/venv/bin/python -m hermes_cli.main --profile work "
            "serve --host 127.0.0.1 --port 0"
        ) == ("work", "/runtime/venv/bin/python")
        assert ui._parse_desktop_serve_command(
            "/runtime/venv/bin/python -m hermes_cli.main -p coder "
            "serve --port=0"
        ) == ("coder", "/runtime/venv/bin/python")

    @pytest.mark.parametrize(
        "command",
        [
            "python -m hermes_cli.main dashboard --port 0",
            "python -m hermes_cli.main serve --port 9119",
            "python -m unrelated.module serve --port 0",
            "notes about hermes_cli.main serve --port 0",
            "python -m hermes_cli.main --profile work chat --port 0",
            "python -m hermes_cli.main --message serve --port 0",
        ],
    )
    def test_rejects_non_desktop_or_non_hermes_shapes(self, command):
        assert ui._parse_desktop_serve_command(command) is None


class TestProcessEnumeration:
    def test_returns_tokenized_rows_and_skips_unreadable_processes(self, monkeypatch):
        class Process:
            def __init__(self, info=None, error=None):
                self._info = info
                self._error = error

            @property
            def info(self):
                if self._error:
                    raise self._error
                return self._info

        expected_argv = [
            "/runtime/venv/bin/python",
            "-m",
            "hermes_cli.main",
            "serve",
            "--port",
            "0",
        ]
        rows = [
            Process({"pid": 501, "cmdline": expected_argv}),
            Process(error=PermissionError("not inspectable")),
            Process({"pid": 502, "cmdline": "not-tokenized"}),
            Process({"pid": os.getpid(), "cmdline": expected_argv}),
        ]
        observed_attrs = []

        def process_iter(attrs):
            observed_attrs.append(attrs)
            return iter(rows)

        monkeypatch.setitem(
            sys.modules,
            "psutil",
            SimpleNamespace(process_iter=process_iter),
        )

        assert ui._iter_process_cmdlines() == [(501, expected_argv)]
        assert observed_attrs == [["pid", "cmdline"]]


class TestPrintPlan:
    def test_git_fleet_output(self, fleet, capsys):
        ui.print_update_plan(ui.collect_runtime_inventory())
        out = capsys.readouterr().out
        assert "Update plan:" in out
        assert "Install: git" in out
        assert "default, work" in out
        assert "pid 100" in out and "systemd" in out
        assert "pid 200" in out and "manual" in out

    def test_docker_warns_not_in_place(self, fleet, monkeypatch, capsys):
        monkeypatch.setattr("hermes_cli.config.detect_install_method", lambda *a, **k: "docker")
        monkeypatch.setattr(
            "hermes_cli.config.recommended_update_command_for_method",
            lambda m: "docker pull nousresearch/hermes-agent:latest",
        )
        ui.print_update_plan(ui.collect_runtime_inventory())
        out = capsys.readouterr().out
        assert "NOT updatable in place" in out
        assert "docker pull" in out

    def test_empty_fleet_message(self, fleet, monkeypatch, capsys):
        monkeypatch.setattr("gateway.status._pid_exists", lambda pid: False)
        ui.print_update_plan(ui.collect_runtime_inventory())
        assert "none detected" in capsys.readouterr().out


class TestReceiptIntegration:
    def test_plan_recorded_into_active_receipt(self, fleet, monkeypatch, tmp_path):
        import hermes_cli.update_receipt as ur

        home = tmp_path / "receipt_home"
        home.mkdir()
        monkeypatch.setattr("hermes_cli.config.get_hermes_home", lambda: home, raising=False)
        ur._current = None
        ur.begin_update_receipt()
        plan = ui.collect_runtime_inventory()
        ui.record_plan_in_receipt(plan)
        path = ur.finalize_update_receipt("success")
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["plan"]["install_method"] == "git"
        assert len(payload["plan"]["runtimes"]) == 2

    def test_noop_without_active_receipt(self, fleet):
        import hermes_cli.update_receipt as ur

        ur._current = None
        ui.record_plan_in_receipt(ui.collect_runtime_inventory())  # must not raise
