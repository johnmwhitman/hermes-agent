"""Runtime inventory + update plan for the fleet-update pipeline.

One read-only pass answering, BEFORE any mutation: which Hermes runtimes run on this machine, how
each is deployed, which ones this update touches, and how each restarts. Every collector is a
side-effect-free probe, so ``hermes update --plan`` is safe on a live fleet.
"""

from __future__ import annotations

import logging
import os
import re
import shlex
from pathlib import Path
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class RuntimeRecord:
    """One running (or expected) Hermes runtime on this machine."""

    kind: str                     # gateway | dashboard | serve
    profile: str
    pid: Optional[int] = None
    supervisor: str = "manual"    # systemd | launchd | desktop | windows-service | service | manual | manual-serve
    code_sha: Optional[str] = None       # stamped running-code sha
    # See #91283.
    code_version: Optional[str] = None
    restart_via: str = ""         # mechanism id, see _RESTART_MECHANISMS
    detail: dict = field(default_factory=dict)


@dataclass
class UpdatePlan:
    """The full pre-update picture: install shape + runtimes + actions."""

    install_method: str = "unknown"       # git | docker | nix | apt | ...
    updatable_in_place: bool = True
    update_mechanism: str = "hermes update"
    expected_sha: Optional[str] = None    # current checkout HEAD (pre-pull)
    expected_version: Optional[str] = None
    profiles: list = field(default_factory=list)
    runtimes: list = field(default_factory=list)  # list[RuntimeRecord]
    inventory_complete: bool = True
    inventory_warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)  # recursive: RuntimeRecord entries become dicts


def _safe_exception_context(exc: BaseException) -> tuple[str, str]:
    """Return diagnostic context without retaining exception payload text."""
    type_name = type(exc).__name__
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", type_name):
        type_name = "Exception"
    code: object = None
    for attribute in ("winerror", "errno", "returncode"):
        candidate = getattr(exc, attribute, None)
        if isinstance(candidate, int) and not isinstance(candidate, bool):
            code = candidate
            break
    return type_name, str(code) if code is not None else "none"

def _log_probe_failure(label: str, exc: BaseException) -> None:
    type_name, code = _safe_exception_context(exc)
    logger.debug("%s [type=%s code=%s]", label, type_name, code)

@dataclass(frozen=True)
class ParsedBackendCommand:
    """Non-secret command shape for a Hermes HTTP backend process."""

    kind: str
    profile: Optional[str]
    port: int
    source_hint: Optional[str] = None

@dataclass(frozen=True)
class ProcessMetadata:
    """Sanitized process metadata retained by the inventory probe."""

    pid: int
    argv: list[str]
    hermes_desktop: bool = False
    hermes_home: Optional[str] = None
    pythonpath: Optional[str] = None

@dataclass
class ProcessScanResult:
    """Process rows plus whether the OS process table was fully readable."""

    rows: list[ProcessMetadata] = field(default_factory=list)
    complete: bool = True
    warnings: list[str] = field(default_factory=list)

def _mark_inventory_incomplete(plan: UpdatePlan, warning: str) -> None:
    """Record sanitized, deduplicated uncertainty from a critical probe."""
    plan.inventory_complete = False
    if warning not in plan.inventory_warnings:
        plan.inventory_warnings.append(warning)

def _canonical_profile_selector(value: str, *, separated: bool) -> Optional[str]:
    """Mirror the side-effect-free value law of ``_apply_profile_override``."""
    try:
        from hermes_cli.profiles import (
            _PROFILE_ID_RE,
            normalize_profile_name,
            validate_profile_name,
        )

        # The pre-parser strictly validates the two-token form before profile
        # normalization, while ``--profile=VALUE`` flows through canonical
        # normalization first. Preserve that measured historical distinction.
        if separated and not _PROFILE_ID_RE.match(value):
            return None
        canonical = normalize_profile_name(value)
        validate_profile_name(canonical)
        return canonical
    except (TypeError, ValueError):
        return None

def _parse_desktop_serve_argv(argv: list[str]) -> Optional[ParsedBackendCommand]:
    """Parse a supported ephemeral Hermes backend command.

    Parsing identifies command shape only.  It deliberately does *not* infer
    Desktop ownership from ``--port 0``; the collector requires independent
    process ownership evidence before assigning the Desktop supervisor.
    """
    if len(argv) < 2:
        return None

    source_hint: Optional[str] = None
    if len(argv) >= 3 and argv[1:3] == ["-m", "hermes_cli.main"]:
        cli_argv = argv[3:]
    elif (
        len(argv) >= 3
        and re.fullmatch(r"python(?:\d+(?:\.\d+)*)?(?:\.exe)?", Path(argv[0]).name, re.I)
        and (
            Path(argv[1]).name.lower() in {"hermes", "hermes.py"}
            or Path(argv[1]).as_posix().lower().endswith("/hermes_cli/main.py")
        )
    ):
        cli_argv = argv[2:]
        script = Path(argv[1]).expanduser()
        if script.is_absolute():
            source_hint = str(
                script.parent.parent
                if script.as_posix().lower().endswith("/hermes_cli/main.py")
                else script.parent
            )
    elif Path(argv[0]).name.lower() in {
        "hermes",
        "hermes.exe",
        "hermes-agent",
        "hermes-agent.exe",
    }:
        cli_argv = argv[1:]
    else:
        return None

    profile: Optional[str] = None
    profile_seen = False
    command_index: Optional[int] = None
    index = 0
    while index < len(cli_argv):
        token = cli_argv[index]
        if token in {"serve", "dashboard"}:
            command_index = index
            break
        if token in {"--profile", "-p"}:
            if profile_seen:
                return None
            if index + 1 >= len(cli_argv):
                return None
            profile = _canonical_profile_selector(
                cli_argv[index + 1], separated=True
            )
            if profile is None:
                return None
            profile_seen = True
            index += 2
            continue
        if token.startswith("--profile="):
            if profile_seen:
                return None
            profile = _canonical_profile_selector(
                token.split("=", 1)[1], separated=False
            )
            if profile is None:
                return None
            profile_seen = True
            index += 1
            continue
        # Desktop only places its optional profile selector before the command.
        # Reject every other token so an option value named ``serve`` cannot
        # masquerade as the subcommand.
        return None

    if command_index is None:
        return None
    port: Optional[str] = None
    tail = cli_argv[command_index + 1 :]
    index = 0
    while index < len(tail):
        token = tail[index]
        if token in {"--profile", "-p"}:
            if profile_seen or index + 1 >= len(tail):
                return None
            profile = _canonical_profile_selector(
                tail[index + 1], separated=True
            )
            if profile is None:
                return None
            profile_seen = True
            index += 2
            continue
        if token.startswith("--profile="):
            if profile_seen:
                return None
            profile = _canonical_profile_selector(
                token.split("=", 1)[1], separated=False
            )
            if profile is None:
                return None
            profile_seen = True
            index += 1
            continue
        if token == "--port":
            if index + 1 >= len(tail):
                return None
            port = tail[index + 1]
            index += 2
            continue
        elif token.startswith("--port="):
            port = token.split("=", 1)[1]
        index += 1
    if port != "0":
        return None
    kind = cli_argv[command_index]
    if kind == "dashboard" and "--no-open" not in tail:
        return None
    return ParsedBackendCommand(
        kind=kind,
        profile=profile,
        port=0,
        source_hint=source_hint,
    )

def _parse_desktop_serve_command(command: str) -> Optional[ParsedBackendCommand]:
    """Parse a display command without retaining credential-bearing argv.

    Desktop owns ephemeral ``serve --port 0`` workers.  Match argv structure,
    not substrings, so an unrelated command that merely mentions Hermes is
    never promoted into the update plan.  The return value deliberately omits
    the rest of argv: serve commands can carry credential-file arguments and
    runtime inventory must never persist them in receipts.
    """
    try:
        argv = shlex.split(command)
    except (TypeError, ValueError):
        return None
    return _parse_desktop_serve_argv(argv)

def _iter_process_cmdlines() -> ProcessScanResult:
    """Read process metadata without retaining arbitrary environment values.

    Any denied or failed process-table read makes the result explicitly
    incomplete.  An update cannot claim convergence from a partial scan.
    """
    result = ProcessScanResult()
    try:
        import psutil

        vanished_types = tuple(
            exception_type
            for exception_type in (
                getattr(psutil, "NoSuchProcess", None),
                getattr(psutil, "ZombieProcess", None),
            )
            if isinstance(exception_type, type)
        )

        for process in psutil.process_iter(["pid", "name"]):
            try:
                info = process.info
                pid = int(info.get("pid"))
                if pid == os.getpid():
                    continue
                process_name = str(info.get("name") or "")
                if not process_name:
                    raise RuntimeError("process name unreadable")
                # Scope privileged argv/environment reads to executable names
                # that can actually host a supported Hermes backend. Denial on
                # one of these candidates is inventory uncertainty; an
                # unrelated kernel/service process is outside this inventory.
                if not (
                    re.fullmatch(
                        r"python(?:w)?(?:\d+(?:\.\d+)*)?(?:\.exe)?",
                        process_name,
                        re.I,
                    )
                    or process_name.lower()
                    in {"hermes", "hermes.exe", "hermes-agent", "hermes-agent.exe"}
                ):
                    continue
                # Call the accessors directly: process_iter(...).info replaces
                # AccessDenied with None, which would turn an uncertain scan
                # into a false "none detected" result.
                raw_argv = process.cmdline() or []
                if not isinstance(raw_argv, (list, tuple)):
                    continue
                argv = [str(token) for token in raw_argv if token is not None]
                if not argv or _parse_desktop_serve_argv(argv) is None:
                    continue
                raw_env = process.environ() or {}
                if not isinstance(raw_env, dict):
                    raw_env = {}
                result.rows.append(
                    ProcessMetadata(
                        pid=pid,
                        argv=argv,
                        hermes_desktop=str(raw_env.get("HERMES_DESKTOP", "")) == "1",
                        hermes_home=(
                            str(raw_env["HERMES_HOME"])
                            if raw_env.get("HERMES_HOME")
                            else None
                        ),
                        pythonpath=(
                            str(raw_env["PYTHONPATH"])
                            if raw_env.get("PYTHONPATH")
                            else None
                        ),
                    )
                )
            except Exception as exc:
                if vanished_types and isinstance(exc, vanished_types):
                    continue
                result.complete = False
                if "one or more process records were unreadable" not in result.warnings:
                    result.warnings.append("one or more process records were unreadable")
                continue
    except Exception as exc:
        _log_probe_failure("Desktop serve process enumeration failed", exc)
        result.complete = False
        result.warnings.append("process inventory unavailable")
    return result

def _hermes_root(path_text: str) -> Optional[Path]:
    """Validate an explicitly process-declared Hermes source root."""
    try:
        path = Path(path_text).expanduser()
        if not path.is_absolute():
            return None
        candidate = path.resolve(strict=False)
        if (candidate / "pyproject.toml").is_file() and (
            candidate / "hermes_cli"
        ).is_dir():
            return candidate
    except (OSError, RuntimeError, ValueError):
        pass
    return None

def _declared_runtime_root(
    process: ProcessMetadata, parsed: ParsedBackendCommand
) -> Optional[Path]:
    """Resolve identity only from a source location declared by the process."""
    for entry in (process.pythonpath or "").split(os.pathsep):
        if entry and (root := _hermes_root(entry)) is not None:
            return root
    if parsed.source_hint:
        return _hermes_root(parsed.source_hint)
    return None

def _code_identity_for_root(root: Optional[Path]) -> dict[str, Optional[str]]:
    """Read non-secret code identity for a discovered runtime root."""
    identity: dict[str, Optional[str]] = {
        "sha": None,
        "version": None,
        "source": None,
    }
    if root is None:
        return identity
    try:
        from hermes_cli.build_info import _resolve_git_head_sha

        identity["sha"] = _resolve_git_head_sha(root)
        if identity["sha"]:
            identity["source"] = "git"
    except Exception:
        pass
    if identity["sha"] is None:
        try:
            baked = (root / ".hermes_build_sha").read_text(
                encoding="utf-8", errors="replace"
            ).strip()
            if baked:
                identity["sha"] = baked
                identity["source"] = "build-file"
        except (OSError, PermissionError):
            pass
    try:
        import tomllib

        with (root / "pyproject.toml").open("rb") as handle:
            raw = tomllib.load(handle).get("project", {}).get("version")
        identity["version"] = str(raw) if raw else None
    except Exception:
        pass
    return identity

def _profile_from_trusted_home(
    hermes_home: Optional[str], profile_homes: list[tuple[str, Path]]
) -> Optional[str]:
    if not hermes_home:
        return None
    try:
        declared = Path(hermes_home).expanduser().resolve(strict=False)
        for profile, home in profile_homes:
            if declared == home.expanduser().resolve(strict=False):
                return profile
    except (OSError, RuntimeError, ValueError):
        pass
    return None

def _collect_desktop_serve_runtimes(
    seen_pids: set[int], profile_homes: list[tuple[str, Path]]
) -> tuple[list[RuntimeRecord], ProcessScanResult]:
    """Discover ephemeral Hermes backends and their proven supervisor."""
    runtimes: list[RuntimeRecord] = []
    try:
        scanned = _iter_process_cmdlines()
    except Exception as exc:
        _log_probe_failure("Desktop serve process scan failed", exc)
        return runtimes, ProcessScanResult(
            complete=False, warnings=["process inventory unavailable"]
        )

    for process in scanned.rows:
        try:
            pid = int(process.pid)
            if pid in seen_pids:
                continue
            parsed = _parse_desktop_serve_argv(process.argv)
            if parsed is None:
                continue
            profile = parsed.profile or _profile_from_trusted_home(
                process.hermes_home, profile_homes
            ) or "unknown"
            supervisor = "desktop" if process.hermes_desktop else "manual"
            identity = _code_identity_for_root(_declared_runtime_root(process, parsed))
            detail: dict[str, str] = {"inventory_source": "process-table"}
            if identity["source"]:
                detail["code_identity_source"] = identity["source"]
            if process.hermes_desktop:
                detail["ownership_evidence"] = "desktop-env"
            runtimes.append(
                RuntimeRecord(
                    kind=parsed.kind,
                    profile=profile,
                    pid=pid,
                    supervisor=supervisor,
                    code_sha=identity["sha"],
                    code_version=identity["version"],
                    restart_via=(
                        _restart_mechanism("desktop", profile)
                        if supervisor == "desktop"
                        else "manual-process"
                    ),
                    detail=detail,
                )
            )
            seen_pids.add(pid)
        except Exception as exc:
            _log_probe_failure("Desktop serve inventory row failed", exc)
            scanned.complete = False
            if "HTTP backend inventory incomplete" not in scanned.warnings:
                scanned.warnings.append("HTTP backend inventory incomplete")
    return runtimes, scanned

def require_complete_inventory(plan: UpdatePlan) -> None:
    """Abort an applying update when runtime convergence cannot be proven."""
    if plan.inventory_complete:
        return
    print()
    print("  ✗ Runtime inventory incomplete; update aborted before mutation.")
    for warning in plan.inventory_warnings:
        print(f"    • {warning}")
    print("    Rerun with process-table visibility, then retry the update.")
    raise SystemExit(1)

def _detect_supervisor_for_pid(pid: int, service_pids: set, windows_service_pids: set | None = None) -> str:
    """Classify how a live gateway PID is supervised."""
    if windows_service_pids and pid in windows_service_pids:
        # SCM-supervised Windows gateway: the update pause machinery stops the SERVICE via sc.exe
        # instead of killing the child, so reconciliation must plan it under its own mechanism id.
        # See #91277.
        return "windows-service"
    if pid not in service_pids:
        return "manual"
    from hermes_cli.gateway import is_macos, supports_systemd_services

    if supports_systemd_services():
        return "systemd"
    if is_macos():
        return "launchd"
    return "service"


# THE restart policy table: restart execution consumes these ids via match_runtime_outcomes / the
# update's restart phase, and the receipt records per-runtime outcomes against them. Display
# strings are derived by describe_restart_mechanism — never the other way around.
_RESTART_MECHANISMS = {
    "systemd": "systemd", "launchd": "launchd", "desktop": "desktop",
    "windows-service": "windows-service", "manual-serve": "respawn-argv",
}

_MECHANISM_DESCRIPTIONS = {
    "systemd": "systemctl restart (drain-first SIGUSR1 when supported)",
    "launchd": "launchctl kickstart -k (drain-first, per-label domain)",
    "desktop": "Desktop app respawns its serve backend",
    "windows-service": "sc.exe stop before venv mutation, sc.exe start after update",
    "respawn-argv": "stop before code swap, relaunch with recorded launch args",
    "manual-process": "restart the Hermes HTTP backend process manually",
}

_SERVE_KINDS = ("serve", "dashboard")


def _restart_mechanism(supervisor: str, profile: str) -> str:
    """Machine-readable restart mechanism id for a runtime.

    THE policy table (#91277 Phase 2): restart execution consumes these ids via
    :func:`match_runtime_outcomes` / the update's restart phase, and the receipt records per-runtime
    outcomes against them. Display strings are derived by :func:`describe_restart_mechanism` — never the
    other way around.
    """
    return _RESTART_MECHANISMS.get(supervisor, "manual")


def describe_restart_mechanism(mechanism: str, profile: str) -> str:
    """Human-readable description of a restart mechanism id."""
    return _MECHANISM_DESCRIPTIONS.get(mechanism) or (
        f"hermes -p {profile} gateway restart" if profile != "default" else "hermes gateway restart"
    )


def _runtime(
    kind: str, profile: str, pid: Optional[int], supervisor: str,
    code_sha: Any = None, code_version: Any = None, **extra: Any,
) -> RuntimeRecord:
    """A :class:`RuntimeRecord` with ``restart_via`` derived from its supervisor."""
    return RuntimeRecord(
        kind=kind, profile=profile, pid=pid, supervisor=supervisor,
        code_sha=str(code_sha) if code_sha else None, code_version=code_version,
        restart_via=_restart_mechanism(supervisor, profile), **extra,
    )


@contextmanager
def _probe(label: str, plan: UpdatePlan):
    """Preserve uncertainty from a failed collector without exposing exception payloads."""
    try:
        yield
    except Exception as exc:
        _log_probe_failure(f"{label} failed", exc)
        _mark_inventory_incomplete(plan, label)


def _collect_install_shape(plan: UpdatePlan) -> None:
    with _probe("install-method inventory unavailable", plan):
        from hermes_cli.config import detect_install_method, get_managed_system, recommended_update_command_for_method

        method = detect_install_method()
        managed = get_managed_system()
        plan.install_method = managed or method
        plan.updatable_in_place = method in ("git", "unknown") and not managed
        # Baked image provenance is authoritative when present: a bind-mounted checkout inside a
        # container can look like `git` while the running filesystem is an immutable image.
        # Fail-closed: an invalid marker still flips the plan to not-updatable.
        with _probe("image provenance inventory unavailable", plan):
            # See #91277.
            from hermes_cli.image_provenance import read_image_provenance

            provenance = read_image_provenance()
            if provenance is not None:
                plan.updatable_in_place = False
                if provenance.valid and provenance.manager:
                    plan.install_method = provenance.manager
        plan.update_mechanism = recommended_update_command_for_method(method)


def _supervisor_classifier(plan: UpdatePlan) -> Callable[[int], str]:
    """Classify supervisors, retaining uncertainty if service ownership cannot be read."""
    service_pids: set = set()
    with _probe("service supervisor inventory unavailable", plan):
        from hermes_cli.gateway import _get_service_pids

        service_pids = _get_service_pids(all_profiles=True) or set()
    # Windows SCM services (no-op off Windows): the update's pause phase stops these via `sc.exe
    # stop` / restarts via `sc.exe start`, so the plan must carry the matching mechanism id.
    # --- SCM-supervised gateway PIDs (Windows) ------------------------------
    # find_windows_gateway_services() maps validated gateway PIDs through process ancestry to running SCM
    # service PIDs (no-op off Windows). See #91277.
    windows_service_pids: set = set()
    with _probe("Windows SCM service inventory unavailable", plan):
        from hermes_cli.gateway import find_windows_gateway_services

        windows_service_pids = {int(service.gateway_pid) for service in find_windows_gateway_services()}
    return lambda pid: _detect_supervisor_for_pid(pid, service_pids, windows_service_pids)


def _collect_gateway_runtimes(plan: UpdatePlan, profile_homes: list, seen: set[int]) -> None:
    """Per-profile gateways: control-socket identity first (declared by the process itself, including
    supervisor provenance — no argv/PID inference), ``gateway_state.json`` fallback, then PID-file
    mapped gateways no status record covers."""
    supervisor = _supervisor_classifier(plan)
    with _probe("gateway runtime-state inventory unavailable", plan):
        from gateway.status import get_runtime_status_running_pid, read_runtime_status
        from gateway.control_socket import identify_gateway

        for profile, home in profile_homes:
            record = None
            with _probe("gateway control-socket inventory unavailable", plan):
                record = identify_gateway(home, strict=True)
            if record is not None:
                pid = int(record["pid"])
                if pid in seen:
                    continue  # one multiplex gateway answers identify for several homes — one record per process
                seen.add(pid)
                declared = record.get("supervisor")
                sup = str(declared) if declared else supervisor(pid)
            else:
                record = read_runtime_status(home / "gateway_state.json", strict=True)
                pid = get_runtime_status_running_pid(record, expected_home=home) if record else None
                if pid is None or pid in seen:
                    continue
                seen.add(pid)
                sup = supervisor(pid)
            plan.runtimes.append(_runtime("gateway", profile, pid, sup, record.get("code_sha"), record.get("code_version")))
    with _probe("gateway PID inventory unavailable", plan):
        from hermes_cli.gateway import find_profile_gateway_processes

        for proc in find_profile_gateway_processes(strict=True):
            if proc.pid not in seen:
                seen.add(proc.pid)
                plan.runtimes.append(_runtime("gateway", proc.profile, proc.pid, supervisor(proc.pid)))


def _collect_ledger_runtimes(plan: UpdatePlan, seen: set[int]) -> None:
    """Serve/dashboard backends from the spawn ledger — runtimes the gateway collectors can never see
    (a manual `hermes serve --host <ip>` for a remote Desktop, a long-lived `hermes dashboard`).
    ledger_entries() live-verifies (pid, create_time) so PID reuse never fabricates a row. Desktop-
    supervised backends (spawner still alive) restart via the Desktop's own respawn, not ours."""
    with _probe("serve/dashboard runtime inventory unavailable", plan):
        from hermes_cli.process_identity import ledger_entries, spawner_is_dead

        for entry in ledger_entries(read_only=True):
            purpose, pid = entry.get("purpose"), entry.get("pid")
            if purpose not in _SERVE_KINDS or not isinstance(pid, int) or pid in seen:
                continue
            seen.add(pid)
            # detail.create_time: process incarnation, not just the numeric PID — a post-update
            # survivor probe comparing PIDs alone calls a NEW serve that reused the number a survivor.
            plan.runtimes.append(_runtime(
                str(purpose), str(entry.get("profile") or "default"), pid,
                "desktop" if spawner_is_dead(entry) is False else "manual-serve",
                detail={
                    "host": entry.get("host") or "",
                    "port": entry.get("port"), "create_time": entry.get("create_time"),
                },
            ))


def collect_runtime_inventory() -> UpdatePlan:
    """Build the pre-update plan. Read-only; never raises — every collector degrades independently.

    The result is embeddable in the update receipt and printable via :func:`print_update_plan`.
    """
    plan = UpdatePlan()
    _collect_install_shape(plan)
    with _probe("code-identity inventory unavailable", plan):
        from hermes_cli.build_info import get_code_identity

        identity = get_code_identity(refresh=True)
        plan.expected_sha = identity.get("sha")
        plan.expected_version = identity.get("version")
    profile_homes: list = []
    with _probe("profile enumeration unavailable", plan):
        from hermes_cli.update_receipt import _profile_homes

        profile_homes = _profile_homes()
        plan.profiles = [name for name, _ in profile_homes]
    seen: set[int] = set()
    _collect_gateway_runtimes(plan, profile_homes, seen)
    _collect_ledger_runtimes(plan, seen)
    # Verified ledger first; older/custom ephemeral backends require a process-table fallback.
    http_runtimes, process_scan = _collect_desktop_serve_runtimes(seen, profile_homes)
    plan.runtimes.extend(http_runtimes)
    if not process_scan.complete:
        for warning in process_scan.warnings or ["process inventory unavailable"]:
            _mark_inventory_incomplete(plan, warning)
    return plan


def print_update_plan(plan: UpdatePlan) -> None:
    """Human-readable plan — what the update will touch and how."""
    print("Update plan:")
    install = f"  Install: {plan.install_method}"
    if plan.expected_version:
        install += f" (v{plan.expected_version}" + (f" @ {plan.expected_sha[:8]}" if plan.expected_sha else "") + ")"
    print(install)
    if not plan.updatable_in_place:
        print("  ⚠ This install is NOT updatable in place.")
        print(f"    Update via: {plan.update_mechanism}")
    print(f"  Profiles: {', '.join(plan.profiles) if plan.profiles else '(none found)'}")
    if not plan.inventory_complete:
        print("  ⚠ Runtime inventory INCOMPLETE; update convergence cannot be proven.")
        for warning in plan.inventory_warnings:
            print(f"    • {warning}")
    if not plan.runtimes:
        if plan.inventory_complete:
            print("  Running Hermes services: none detected — code swap only.")
        else:
            print("  Running Hermes services: no readable runtimes found (scan incomplete).")
        return
    print(f"  Running services to restart ({len(plan.runtimes)}):")
    for runtime in plan.runtimes:
        sha = f" @ {runtime.code_sha[:8]}" if runtime.code_sha else ""
        print(f"    • {runtime.kind} [{runtime.profile}] pid {runtime.pid} — {runtime.supervisor}{sha}")
        print(f"      restart: {describe_restart_mechanism(runtime.restart_via, runtime.profile)}")


def _serve_unit_matches_profile(profile: str, unit: object) -> bool:
    """Does *unit* name a ``hermes-serve*``/``hermes-dashboard*`` unit for *profile*? (OWN vocabulary;
    the gateway's ``hermes-gateway*`` names never cover serve/dashboard runtimes.)

    Exact names only — ``work`` must not claim ``hermes-serve-workbench`` — and a scope prefix
    (``user/hermes-serve``) is tolerated because the restart phase records scope-qualified identities in
    some lists. See #100479.
    """
    name = str(unit).removesuffix(".service").rsplit("/", 1)[-1]
    suffix = "" if profile == "default" else f"-{profile}"
    return name in {f"hermes-serve{suffix}", f"hermes-dashboard{suffix}"}


def _gateway_service_matches_profile(profile: str, service: object) -> bool:
    """Match an exact gateway service/label (systemd/launchd/s6 shapes) to a profile.

    Never substring-match: ``foo`` must not claim ``hermes-gateway-foobar.service``.
    Launchd labels are ``ai.hermes.gateway`` / ``ai.hermes.gateway-<profile>`` — they do
    not contain the substring ``hermes-gateway``, so a successful macOS kickstart must
    still credit the planned default gateway. A scope prefix (``user/hermes-gateway``,
    ``gui/501/ai.hermes.gateway``) is stripped the same way serve units are.
    """
    name = str(service).removesuffix(".service").rsplit("/", 1)[-1]
    if profile == "default":
        return name in {"hermes-gateway", "ai.hermes.gateway", "gateway", "gateway-default"}
    return name in {f"hermes-gateway-{profile}", f"ai.hermes.gateway-{profile}", f"gateway-{profile}"}


def _gateway_named_in(r: RuntimeRecord, names: set) -> bool:
    # Gateway-only vocabulary: a serve/dashboard that merely shares the profile is a
    # different process. Exact label match (systemd + launchd + s6), not substring.
    return any(_gateway_service_matches_profile(r.profile, name) for name in names)


def match_runtime_outcomes(
    plan: "UpdatePlan", *, restarted_services: list, relaunched_profiles: list,
    externally_supervised_profiles: list, killed_pids: set, failed_units: list,
    stale_serve_pids: "set | None" = None,
) -> list[dict[str, Any]]:
    """Reconcile the plan's runtimes against what the restart phase DID.

    The platform restart branches each re-discover their own targets, so a runtime the plan saw can
    be missed with no signal. Returns one ``{kind, profile, pid, mechanism, outcome}`` row per
    planned runtime; outcome is ``restarted``, ``stopped``, ``failed`` or ``unaccounted`` (no
    bookkeeping mentions it — the blind-spot tripwire). Never raises. Serve/dashboard runtimes are
    reconciled in their OWN vocabulary and never borrow the gateway's outcome: with
    ``stale_serve_pids`` a pre-update serve whose incarnation is gone counts as ``restarted``, one
    still alive is ``unaccounted``; without the probe an untouched serve stays ``unaccounted``.

    See #91277.
    They never borrow the gateway's outcome: ``relaunched_profiles`` and ``hermes-gateway*`` name a
    different process that shares the profile, nothing more. See #100479.
    """
    outcomes: list[dict[str, Any]] = []
    try:
        failed_set = {str(u) for u in (failed_units or [])}
        restarted_set = {str(s) for s in (restarted_services or [])}
        relaunched = set(relaunched_profiles or []) | set(externally_supervised_profiles or [])
        killed = {int(p) for p in (killed_pids or set())}
        stale_serves = {int(p) for p in stale_serve_pids} if stale_serve_pids is not None else None

        def _outcome(r: RuntimeRecord) -> str:
            killed_here = r.pid is not None and r.pid in killed
            if r.kind in _SERVE_KINDS:
                if killed_here:
                    return "stopped"
                if any(_serve_unit_matches_profile(r.profile, u) for u in failed_set):
                    return "failed"
                if stale_serves is not None:
                    # Incarnation-verified: the pre-update process is gone (replaced by its unit / the
                    # dashboard cleanup respawn / the Desktop app) or it is still alive on pre-update code.
                    return "unaccounted" if r.pid in stale_serves else "restarted"
                return "restarted" if any(_serve_unit_matches_profile(r.profile, s) for s in restarted_set) else "unaccounted"
            if r.profile in relaunched:
                return "restarted"
            if killed_here:
                return "stopped"
            if _gateway_named_in(r, failed_set):
                return "failed"
            return "restarted" if _gateway_named_in(r, restarted_set) else "unaccounted"

        for r in plan.runtimes:
            if isinstance(r, RuntimeRecord):
                outcomes.append(
                    {"kind": r.kind, "profile": r.profile, "pid": r.pid, "mechanism": r.restart_via, "outcome": _outcome(r)}
                )
    except Exception as exc:
        _log_probe_failure("Runtime-outcome reconciliation failed", exc)
        _mark_inventory_incomplete(plan, "runtime-outcome reconciliation unavailable")
    if not plan.inventory_complete:
        outcomes.append({
            "kind": "inventory", "profile": "unknown", "pid": None,
            "mechanism": "process-scan", "outcome": "unaccounted",
        })
    return outcomes


def report_unaccounted_runtimes(outcomes: list[dict[str, Any]]) -> bool:
    """Print a loud warning for runtimes the restart phase never touched.

    Returns True when at least one planned runtime is unaccounted; the caller escalates like a
    STALE/DOWN fleet row (exit 1) — a promised restart silently missed is the class this phase
    exists to kill.
    """
    missed = [o for o in outcomes if o.get("outcome") == "unaccounted"]
    if not missed:
        return False
    print()
    print("  ⚠ Planned runtimes the restart phase never touched:")
    for o in missed:
        print(f"    ✗ {o['kind']} [{o['profile']}] pid {o['pid']} — planned mechanism: {o['mechanism']}")
    print("    Restart them manually, then verify:")
    if any(o.get("kind") == "inventory" for o in missed):
        print("      rerun the inventory with process-table visibility")
    if any(o.get("kind") == "gateway" for o in missed):
        print("      hermes gateway restart                # active profile")
        print("      hermes -p <profile> gateway restart   # named profile")
    if any(o.get("kind") in _SERVE_KINDS for o in missed):
        # A serve/dashboard is not reachable by any `gateway restart` command: name the process, not the wrong verb.
        # See #100479.
        print("      systemctl --user restart hermes-serve.service   # unit-managed serve")
        print("      relaunch `hermes serve` / `hermes dashboard` / the Desktop app")
    return True


def record_plan_in_receipt(plan: UpdatePlan) -> None:
    """Attach the inventory to the active update receipt. Never raises."""
    try:
        import hermes_cli.update_receipt as ur

        if ur._current is not None:
            ur._current.data["plan"] = plan.to_dict()
    except Exception as exc:
        _log_probe_failure("Could not record plan in receipt", exc)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.
# ---- END PLUGIN-COMPAT ----
