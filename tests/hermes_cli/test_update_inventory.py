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
    monkeypatch.setattr(
        "hermes_cli.gateway.find_windows_gateway_services", lambda: []
    )
    monkeypatch.setattr("hermes_cli.gateway.supports_systemd_services", lambda: True)
    monkeypatch.setattr("hermes_cli.gateway.find_profile_gateway_processes", lambda exclude_pids=None: [])
    monkeypatch.setattr("hermes_cli.process_identity.ledger_entries", lambda: [])
    monkeypatch.setattr(
        ui,
        "_iter_process_cmdlines",
        lambda: ui.ProcessScanResult(),
        raising=False,
    )
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
            "hermes_cli.gateway.find_windows_gateway_services",
            "hermes_cli.gateway.find_profile_gateway_processes",
            "hermes_cli.process_identity.ledger_entries",
        ):
            monkeypatch.setattr(target, _boom)
        monkeypatch.setattr(ui, "_iter_process_cmdlines", _boom)
        plan = ui.collect_runtime_inventory()
        assert plan.runtimes == []
        assert plan.install_method == "unknown"
        assert plan.inventory_complete is False

    @pytest.mark.parametrize(
        ("target", "warning"),
        [
            ("hermes_cli.profiles._get_default_hermes_home", "profile enumeration unavailable"),
            ("hermes_cli.gateway._get_service_pids", "service supervisor inventory unavailable"),
            (
                "hermes_cli.gateway.find_windows_gateway_services",
                "Windows SCM service inventory unavailable",
            ),
            ("hermes_cli.gateway.supports_systemd_services", "gateway runtime-state inventory unavailable"),
            ("gateway.status.read_runtime_status", "gateway runtime-state inventory unavailable"),
            ("hermes_cli.gateway.find_profile_gateway_processes", "gateway PID inventory unavailable"),
            (
                "hermes_cli.process_identity.ledger_entries",
                "serve/dashboard runtime inventory unavailable",
            ),
        ],
    )
    def test_critical_runtime_probe_failure_marks_inventory_incomplete(
        self, fleet, monkeypatch, target, warning
    ):
        def denied(*_args, **_kwargs):
            raise PermissionError("sensitive host detail must not be reported")

        monkeypatch.setattr(target, denied)
        plan = ui.collect_runtime_inventory()
        assert plan.inventory_complete is False
        assert warning in plan.inventory_warnings
        assert "sensitive host detail" not in json.dumps(plan.to_dict())

    def test_desktop_row_failure_marks_process_inventory_incomplete(
        self, fleet, monkeypatch
    ):
        monkeypatch.setattr(
            ui,
            "_iter_process_cmdlines",
            lambda: ui.ProcessScanResult(
                rows=[
                    ui.ProcessMetadata(
                        pid="not-a-pid",
                        argv=["python", "-m", "hermes_cli.main", "serve", "--port", "0"],
                        hermes_desktop=True,
                    )
                ]
            ),
        )
        plan = ui.collect_runtime_inventory()
        assert plan.inventory_complete is False
        assert "HTTP backend inventory incomplete" in plan.inventory_warnings

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
        runtime_root = tmp_path / "desktop-worktree"
        main_runtime = tmp_path / "main-runtime"
        python = main_runtime / "venv" / "bin" / "python"
        python.parent.mkdir(parents=True)
        (runtime_root / "hermes_cli").mkdir(parents=True)
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
            lambda: ui.ProcessScanResult(
                rows=[
                    ui.ProcessMetadata(
                        pid=303,
                        argv=[
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
                        hermes_desktop=True,
                        hermes_home=str(tmp_path / "home" / "profiles" / "work"),
                        pythonpath=str(runtime_root),
                    )
                ]
            ),
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
        assert serve.detail["code_identity_source"] == "git"
        assert "desktop-env" in serve.detail["ownership_evidence"]
        serialized = json.dumps(serve.to_dict())
        assert secret_path not in serialized
        assert "ssh-session-token-file" not in serialized

    def test_desktop_serve_scan_is_strict_and_deduplicates_gateway_pid(
        self, fleet, monkeypatch
    ):
        monkeypatch.setattr(
            ui,
            "_iter_process_cmdlines",
            lambda: ui.ProcessScanResult(
                rows=[
                    ui.ProcessMetadata(100, ["python", "-m", "hermes_cli.main", "serve", "--port", "0"], True),
                    ui.ProcessMetadata(301, ["python", "-m", "hermes_cli.main", "dashboard", "--port", "0"], True),
                    ui.ProcessMetadata(302, ["python", "-m", "hermes_cli.main", "serve", "--port", "9119"], True),
                    ui.ProcessMetadata(303, ["notes", "about", "hermes_cli.main", "serve", "--port", "0"], True),
                    ui.ProcessMetadata(304, ["python", "-m", "unrelated.module", "serve", "--port", "0"], True),
                ]
            ),
            raising=False,
        )

        plan = ui.collect_runtime_inventory()

        assert [r.pid for r in plan.runtimes].count(100) == 1
        assert not ({301, 302, 303, 304} & {r.pid for r in plan.runtimes})

    def test_bare_profile_comes_only_from_trusted_hermes_home(self, fleet, monkeypatch):
        monkeypatch.setattr(
            ui,
            "_iter_process_cmdlines",
            lambda: ui.ProcessScanResult(
                rows=[
                    ui.ProcessMetadata(
                        601,
                        ["python", "-m", "hermes_cli.main", "serve", "--port", "0"],
                        True,
                        hermes_home=str(fleet / "home" / "profiles" / "work"),
                    ),
                    ui.ProcessMetadata(
                        602,
                        ["python", "-m", "hermes_cli.main", "serve", "--port", "0"],
                        True,
                        hermes_home=str(fleet / "untrusted-home"),
                    ),
                ]
            ),
        )
        plan = ui.collect_runtime_inventory()
        by_pid = {runtime.pid: runtime for runtime in plan.runtimes}
        assert by_pid[601].profile == "work"
        assert by_pid[602].profile == "unknown"

    def test_port_zero_without_desktop_ownership_stays_manual(self, fleet, monkeypatch):
        monkeypatch.setattr(
            ui,
            "_iter_process_cmdlines",
            lambda: ui.ProcessScanResult(
                rows=[
                    ui.ProcessMetadata(
                        603,
                        [
                            "python",
                            "-m",
                            "hermes_cli.main",
                            "--profile",
                            "work",
                            "serve",
                            "--host",
                            "0.0.0.0",
                            "--port",
                            "0",
                        ],
                    )
                ]
            ),
        )
        runtime = next(r for r in ui.collect_runtime_inventory().runtimes if r.pid == 603)
        assert runtime.supervisor == "manual"
        assert runtime.restart_via == "manual-process"

    def test_legacy_dashboard_and_python_script_launch_are_discovered(
        self, fleet, monkeypatch, tmp_path
    ):
        runtime_root = tmp_path / "remote-runtime"
        (runtime_root / "hermes_cli").mkdir(parents=True)
        (runtime_root / "pyproject.toml").write_text(
            '[project]\nversion="6.5.4"\n', encoding="utf-8"
        )
        (runtime_root / ".hermes_build_sha").write_text(
            "f" * 40, encoding="utf-8"
        )
        monkeypatch.setattr(
            ui,
            "_iter_process_cmdlines",
            lambda: ui.ProcessScanResult(
                rows=[
                    ui.ProcessMetadata(
                        604,
                        [
                            "python",
                            str(runtime_root / "hermes"),
                            "--profile",
                            "work",
                            "dashboard",
                            "--no-open",
                            "--port",
                            "0",
                        ],
                        True,
                    )
                ]
            ),
        )
        runtime = next(r for r in ui.collect_runtime_inventory().runtimes if r.pid == 604)
        assert runtime.kind == "dashboard"
        assert runtime.supervisor == "desktop"
        assert runtime.code_sha == "f" * 40
        assert runtime.code_version == "6.5.4"
        assert runtime.detail["code_identity_source"] == "build-file"

    def test_unknown_source_provenance_does_not_use_python_venv(self, fleet, monkeypatch, tmp_path):
        runtime_root = tmp_path / "runtime"
        python = runtime_root / "venv" / "bin" / "python"
        python.parent.mkdir(parents=True)
        (runtime_root / "hermes_cli").mkdir()
        (runtime_root / "pyproject.toml").write_text('[project]\nversion="7.7.7"\n')
        (runtime_root / ".hermes_build_sha").write_text("e" * 40)
        monkeypatch.setattr(
            ui,
            "_iter_process_cmdlines",
            lambda: ui.ProcessScanResult(
                rows=[ui.ProcessMetadata(605, [str(python), "-m", "hermes_cli.main", "serve", "--port", "0"], True)]
            ),
        )
        runtime = next(r for r in ui.collect_runtime_inventory().runtimes if r.pid == 605)
        assert runtime.code_sha is None
        assert runtime.code_version is None


class TestDesktopServeCommandParser:
    def test_accepts_module_entrypoint_and_profile_flag_forms(self):
        assert ui._parse_desktop_serve_command(
            "/runtime/venv/bin/python -m hermes_cli.main --profile work "
            "serve --host 127.0.0.1 --port 0"
        ).profile == "work"
        assert ui._parse_desktop_serve_command(
            "/runtime/venv/bin/python -m hermes_cli.main -p coder "
            "serve --port=0"
        ).profile == "coder"

    def test_bare_profile_is_unknown_and_duplicate_profile_flags_are_rejected(self):
        bare = ui._parse_desktop_serve_command(
            "python -m hermes_cli.main serve --port 0"
        )
        assert bare.profile is None
        assert ui._parse_desktop_serve_command(
            "python -m hermes_cli.main --profile first -p second serve "
            "--port 9119 --port=0"
        ) is None

    @pytest.mark.parametrize(
        ("command", "profile", "kind"),
        [
            (
                "python -m hermes_cli.main serve --profile default --port 0 --status",
                "default",
                "serve",
            ),
            (
                "python -m hermes_cli.main dashboard --no-open -p work --port 0",
                "work",
                "dashboard",
            ),
        ],
    )
    def test_accepts_one_profile_selector_after_backend_subcommand(
        self, command, profile, kind
    ):
        parsed = ui._parse_desktop_serve_command(command)
        assert parsed.profile == profile
        assert parsed.kind == kind

    @pytest.mark.parametrize(
        "command",
        [
            "python -m hermes_cli.main --profile first serve -p second --port 0",
            "python -m hermes_cli.main serve --profile first -p second --port 0",
            "python -m hermes_cli.main --profile first --profile second serve --port 0",
        ],
    )
    def test_rejects_duplicate_profile_selectors_in_any_position(self, command):
        assert ui._parse_desktop_serve_command(command) is None

    @pytest.mark.parametrize(
        "command",
        [
            "python -m hermes_cli.main serve --profile --port 0",
            "python -m hermes_cli.main serve -p --port 0",
            "python -m hermes_cli.main serve --profile= --port 0",
        ],
    )
    def test_rejects_missing_or_invalid_tail_profile_value(self, command):
        assert ui._parse_desktop_serve_command(command) is None

    def test_profile_value_semantics_match_preparser(self):
        assert ui._parse_desktop_serve_command(
            "python -m hermes_cli.main serve --profile DEFAULT --port 0"
        ) is None

        equals_default = ui._parse_desktop_serve_command(
            "python -m hermes_cli.main serve --profile=DEFAULT --port 0"
        )
        assert equals_default.profile == "default"

        assert ui._parse_desktop_serve_command(
            "python -m hermes_cli.main serve --profile root --port 0"
        ) is None
        assert ui._parse_desktop_serve_command(
            "python -m hermes_cli.main serve --profile=ROOT --port 0"
        ) is None
        assert ui._parse_desktop_serve_command(
            "python -m hermes_cli.main serve --profile " + ("a" * 65) + " --port 0"
        ) is None
        assert ui._parse_desktop_serve_command(
            "python -m hermes_cli.main serve --profile=" + ("a" * 65) + " --port 0"
        ) is None

    def test_accepts_legacy_dashboard_and_remote_python_script_family(self):
        parsed = ui._parse_desktop_serve_command(
            "python /opt/hermes/hermes --profile remote dashboard --no-open --port 0"
        )
        assert parsed.kind == "dashboard"
        assert parsed.profile == "remote"
        assert parsed.source_hint == "/opt/hermes"

    def test_rejects_trailing_duplicate_port_without_value(self):
        assert ui._parse_desktop_serve_command(
            "python -m hermes_cli.main serve --port 0 --port"
        ) is None

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
    def test_returns_sanitized_rows_and_marks_denial_incomplete(self, monkeypatch):
        class Process:
            def __init__(self, info=None, error=None):
                self._info = info
                self._error = error

            @property
            def info(self):
                if not self._info:
                    return {}
                argv = self._info.get("cmdline")
                name = Path(argv[0]).name if isinstance(argv, list) and argv else "other"
                return {"pid": self._info.get("pid"), "name": name}

            def cmdline(self):
                if self._error:
                    raise self._error
                return self._info.get("cmdline")

            def environ(self):
                return self._info.get("environ") or {}

        expected_argv = [
            "/runtime/venv/bin/python",
            "-m",
            "hermes_cli.main",
            "serve",
            "--port",
            "0",
        ]
        rows = [
            Process({"pid": 501, "cmdline": expected_argv, "environ": {
                "HERMES_DESKTOP": "1",
                "HERMES_HOME": "/profiles/work",
                "PYTHONPATH": "/worktree",
                "HERMES_DASHBOARD_SESSION_TOKEN": "never-return-me",
            }}),
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

        result = ui._iter_process_cmdlines()
        assert result.complete is False
        assert len(result.warnings) == 1
        assert result.rows == [
            ui.ProcessMetadata(
                501,
                expected_argv,
                True,
                hermes_home="/profiles/work",
                pythonpath="/worktree",
            )
        ]
        assert "never-return-me" not in repr(result)
        assert observed_attrs == [["pid", "name"]]

    def test_iterator_failure_is_explicitly_incomplete(self, monkeypatch):
        def fail(_attrs):
            raise PermissionError("process table denied")

        monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace(process_iter=fail))
        result = ui._iter_process_cmdlines()
        assert result.rows == []
        assert result.complete is False
        assert result.warnings == ["process inventory unavailable"]

    @pytest.mark.parametrize("exception_name", ["NoSuchProcess", "ZombieProcess"])
    def test_vanished_process_race_is_skipped_without_poisoning_inventory(
        self, monkeypatch, exception_name
    ):
        class NoSuchProcess(Exception):
            pass

        class ZombieProcess(Exception):
            pass

        exception_type = {
            "NoSuchProcess": NoSuchProcess,
            "ZombieProcess": ZombieProcess,
        }[exception_name]

        class Process:
            info = {"pid": 801, "name": "python"}

            def cmdline(self):
                raise exception_type("process exited")

        monkeypatch.setitem(
            sys.modules,
            "psutil",
            SimpleNamespace(
                process_iter=lambda _attrs: iter([Process()]),
                NoSuchProcess=NoSuchProcess,
                ZombieProcess=ZombieProcess,
            ),
        )
        result = ui._iter_process_cmdlines()
        assert result.rows == []
        assert result.complete is True
        assert result.warnings == []

    def test_access_denied_candidate_stays_fail_closed_with_static_warning(
        self, monkeypatch
    ):
        class AccessDenied(Exception):
            pass

        class Process:
            info = {"pid": 802, "name": "python"}

            def cmdline(self):
                raise AccessDenied("private detail")

        monkeypatch.setitem(
            sys.modules,
            "psutil",
            SimpleNamespace(
                process_iter=lambda _attrs: iter([Process()]),
                NoSuchProcess=type("NoSuchProcess", (Exception,), {}),
                ZombieProcess=type("ZombieProcess", (Exception,), {}),
                AccessDenied=AccessDenied,
            ),
        )
        result = ui._iter_process_cmdlines()
        assert result.complete is False
        assert result.warnings == ["one or more process records were unreadable"]
        assert "private detail" not in repr(result)


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

    def test_incomplete_empty_scan_never_claims_none_detected(self, fleet, monkeypatch, capsys):
        monkeypatch.setattr("gateway.status._pid_exists", lambda pid: False)
        monkeypatch.setattr(
            ui,
            "_iter_process_cmdlines",
            lambda: ui.ProcessScanResult(complete=False, warnings=["process inventory unavailable"]),
        )
        plan = ui.collect_runtime_inventory()
        ui.print_update_plan(plan)
        out = capsys.readouterr().out
        assert "INCOMPLETE" in out
        assert "convergence cannot be proven" in out
        assert "none detected" not in out

    def test_incomplete_inventory_gate_is_loud_and_aborts_before_mutation(
        self, capsys
    ):
        plan = ui.UpdatePlan(
            inventory_complete=False,
            inventory_warnings=["process inventory unavailable"],
        )
        with pytest.raises(SystemExit) as raised:
            ui.require_complete_inventory(plan)
        assert raised.value.code == 1
        out = capsys.readouterr().out
        assert "aborted before mutation" in out
        assert "process inventory unavailable" in out

    def test_complete_inventory_gate_is_noop(self, capsys):
        ui.require_complete_inventory(ui.UpdatePlan())
        assert capsys.readouterr().out == ""


class TestRuntimeOutcomeIsolation:
    @staticmethod
    def _outcomes(plan, **overrides):
        args = {
            "restarted_services": [],
            "relaunched_profiles": [],
            "externally_supervised_profiles": [],
            "killed_pids": set(),
            "failed_units": [],
        }
        args.update(overrides)
        return ui.match_runtime_outcomes(plan, **args)

    def test_gateway_restart_cannot_satisfy_same_profile_serve(self):
        plan = ui.UpdatePlan(
            runtimes=[
                ui.RuntimeRecord(
                    kind="serve",
                    profile="default",
                    pid=701,
                    supervisor="desktop",
                    restart_via="desktop",
                )
            ]
        )
        for evidence in (
            {"restarted_services": ["hermes-gateway.service"]},
            {"relaunched_profiles": ["default"]},
            {"externally_supervised_profiles": ["default"]},
        ):
            assert self._outcomes(plan, **evidence)[0]["outcome"] == "unaccounted"

    def test_incomplete_scan_adds_unaccounted_inventory_tripwire(self):
        plan = ui.UpdatePlan(
            inventory_complete=False,
            inventory_warnings=["process inventory unavailable"],
        )
        outcomes = self._outcomes(plan)
        assert outcomes == [
            {
                "kind": "inventory",
                "profile": "unknown",
                "pid": None,
                "mechanism": "process-scan",
                "outcome": "unaccounted",
            }
        ]
        assert ui.report_unaccounted_runtimes(outcomes) is True


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
