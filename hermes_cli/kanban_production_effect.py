"""Production-effect field + close-time gate (additive; unused until applied).

Mirrors hermes_cli.kanban_pr_acceptance: a small validator + receipt dict,
no network, no writes. Callers (create_task / complete_task / CYCLE-SUMMARY /
dashboards) stay on the old path when the column is absent, the value is
NULL/internal, or HERMES_KANBAN_PRODUCTION_EFFECT is off.

This file is the proposed hermes_cli/kanban_production_effect.py. It is
NOT installed in the live Hermes checkout.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
from typing import Any, Callable, Mapping, Optional

VALID_EFFECTS = frozenset({"production", "enabling", "internal"})
DEFAULT_EFFECT = "internal"
COLUMN_DDL = "production_effect TEXT"
ENV_FLAG = "HERMES_KANBAN_PRODUCTION_EFFECT"
KILL_SWITCH_OFF = frozenset({"0", "off", "false", "no"})

_TASK_ID_RE = re.compile(r"^t_[0-9a-f]{8,}$")
_UNBLOCKS_RE = re.compile(r"\bunblocks[:\s]+(t_[0-9a-f]{8,})\b", re.I)
_DEPLOY_ID_RE = re.compile(r"\bdeploy[_-]?id[:\s]+(\S+)", re.I)
_RELEASE_TAG_RE = re.compile(r"\brelease[_-]?tag[:\s]+(\S+)", re.I)
_LIVE_URL_RE = re.compile(r"\blive[_-]?url[:\s]+(https?://\S+)", re.I)
_HTTP_STATUS_RE = re.compile(r"\bHTTP[:\s]+([1-3]\d{2})\b", re.I)
_URL_RE = re.compile(r"https?://[^\s'\"<>]+")
_ARTIFACT_KINDS = frozenset({"deploy_id", "release_tag", "live_url", "phase05_receipt"})


def validate_production_effect(value: str | None) -> str | None:
    """Persist NULL for omitted/internal so existing INSERT shape stays valid.

    Explicit ``internal`` is stored as ``internal``. None/"" stay None
    (readers treat NULL as internal). Invalid values raise, matching
    ``kanban_pr_acceptance.validate_contract``.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("production_effect must be production|enabling|internal")
    stripped = value.strip().lower()
    if stripped == "":
        return None
    if stripped not in VALID_EFFECTS:
        raise ValueError("production_effect must be production|enabling|internal")
    return stripped


def classify_effect(value: str | None) -> str:
    """NULL/unknown → internal. Does not raise (dashboard/CYCLE-SUMMARY path)."""
    if not value:
        return DEFAULT_EFFECT
    stripped = str(value).strip().lower()
    return stripped if stripped in VALID_EFFECTS else DEFAULT_EFFECT


def enforcement_enabled(environ: Mapping[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    raw = str(env.get(ENV_FLAG, "on")).strip().lower()
    return raw not in KILL_SWITCH_OFF


def _prose(summary: str, result: str) -> str:
    return "\n".join(part for part in (summary or "", result or "") if part)


def parse_unblocks(text: str, metadata: Mapping[str, Any] | None) -> str | None:
    meta = metadata if isinstance(metadata, Mapping) else {}
    raw = meta.get("unblocks")
    if isinstance(raw, str) and _TASK_ID_RE.fullmatch(raw.strip()):
        return raw.strip()
    match = _UNBLOCKS_RE.search(text or "")
    return match.group(1) if match else None


def parse_production_artifact(
    text: str, metadata: Mapping[str, Any] | None
) -> dict[str, Any] | None:
    """Return {kind, value, observed_status?} or None.

    Prefer metadata.production_artifact; fall back to receipt-grammar-shaped
    prose tokens (deploy_id: / release_tag: / live_url: + HTTP 2xx/3xx).
    """
    meta = metadata if isinstance(metadata, Mapping) else {}
    raw = meta.get("production_artifact")
    if isinstance(raw, Mapping):
        kind = str(raw.get("kind") or "").strip().lower()
        value = raw.get("value")
        if kind in _ARTIFACT_KINDS and isinstance(value, str) and value.strip():
            artifact: dict[str, Any] = {"kind": kind, "value": value.strip()}
            status = raw.get("observed_status")
            if status is not None:
                artifact["observed_status"] = int(status)
            return artifact
    text = text or ""
    deploy = _DEPLOY_ID_RE.search(text)
    if deploy:
        return {"kind": "deploy_id", "value": deploy.group(1).rstrip(".,;:")}
    tag = _RELEASE_TAG_RE.search(text)
    if tag:
        return {"kind": "release_tag", "value": tag.group(1).rstrip(".,;:")}
    live = _LIVE_URL_RE.search(text)
    if live:
        artifact = {"kind": "live_url", "value": live.group(1).rstrip(".,;:")}
        status_match = _HTTP_STATUS_RE.search(text)
        if status_match:
            artifact["observed_status"] = int(status_match.group(1))
        return artifact
    return None


def phase05_published(path: str) -> bool:
    """True when path is a phase05-receipt/v1 envelope with state=PUBLISHED.

    Reuses SUCCESSION/RECEIPT-GRAMMAR.v1.md terminals without calling
    lint-receipt.py (that linter stays the YourBrief pre-commit gate).
    """
    try:
        with open(path, encoding="utf-8") as handle:
            envelope = json.load(handle)
    except (OSError, json.JSONDecodeError, UnicodeError):
        return False
    if not isinstance(envelope, dict):
        return False
    return (
        envelope.get("schema") == "phase05-receipt/v1"
        and envelope.get("state") == "PUBLISHED"
        and isinstance(envelope.get("craft"), dict)
        and isinstance((envelope.get("craft") or {}).get("user_visible_change"), str)
        and len(str(envelope["craft"]["user_visible_change"]).strip()) >= 8
    )


def _artifact_ok(artifact: dict[str, Any]) -> tuple[bool, str]:
    kind = artifact.get("kind")
    value = artifact.get("value")
    if not isinstance(value, str) or not value.strip():
        return False, "production_artifact.value is empty"
    if kind == "deploy_id":
        return True, "deploy_id"
    if kind == "release_tag":
        return True, "release_tag"
    if kind == "live_url":
        if not _URL_RE.fullmatch(value.rstrip(".,;:")):
            return False, "live_url value is not an http(s) URL"
        status = artifact.get("observed_status")
        if not isinstance(status, int) or not (200 <= status < 400):
            return False, "live_url requires observed_status 200-399 (or HTTP 2xx/3xx in prose)"
        return True, "live_url"
    if kind == "phase05_receipt":
        if not phase05_published(value):
            return False, "phase05_receipt is not schema=phase05-receipt/v1 state=PUBLISHED"
        return True, "phase05_receipt"
    return False, f"unknown production_artifact.kind {kind!r}"


LookupTask = Callable[[str], Optional[Mapping[str, Any]]]


def enforce_on_complete(
    effect: str | None,
    *,
    summary: str = "",
    result: str = "",
    metadata: Mapping[str, Any] | None = None,
    artifacts: list[str] | None = None,
    lookup_task: LookupTask | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Return a receipt dict. ok=False means complete_task must refuse.

    Old path (ok=True, classification=skipped): kill switch off, or
    classified effect is internal. Does not replace the existing
    _enforce_receipt_on_complete / PR-acceptance gates.
    """
    receipt: dict[str, Any] = {
        "ok": True,
        "classification": "skipped",
        "effect": classify_effect(effect),
        "detail": None,
        "recovery": (
            "Fix the production_effect receipt and retry kanban_complete. "
            "Use HERMES_KANBAN_PRODUCTION_EFFECT=off only to roll back the gate."
        ),
    }
    if not enforcement_enabled(environ):
        receipt["classification"] = "disabled"
        return receipt
    classified = classify_effect(effect)
    receipt["effect"] = classified
    if classified == "internal":
        return receipt

    text = _prose(summary, result)
    meta = metadata if isinstance(metadata, Mapping) else {}

    if classified == "production":
        artifact = parse_production_artifact(text, meta)
        if artifact is None and artifacts:
            for path in artifacts:
                if isinstance(path, str) and path.endswith(".json") and phase05_published(path):
                    artifact = {"kind": "phase05_receipt", "value": path}
                    break
        if artifact is None:
            receipt.update(
                ok=False,
                classification="missing",
                detail=(
                    "production card done-receipt must cite a production artifact: "
                    "metadata.production_artifact {kind: deploy_id|release_tag|live_url|phase05_receipt, value} "
                    "or prose deploy_id: / release_tag: / live_url: + HTTP 2xx/3xx."
                ),
            )
            return receipt
        ok, why = _artifact_ok(artifact)
        receipt["artifact"] = artifact
        if not ok:
            receipt.update(ok=False, classification="invalid", detail=why)
            return receipt
        receipt["classification"] = why
        return receipt

    # enabling
    unblocks = parse_unblocks(text, meta)
    if unblocks is None:
        receipt.update(
            ok=False,
            classification="missing",
            detail=(
                "enabling card must name the production card it unblocks via "
                "metadata.unblocks=t_<hex> or prose 'unblocks: t_<hex>'."
            ),
        )
        return receipt
    receipt["unblocks"] = unblocks
    if lookup_task is None:
        receipt.update(
            ok=False,
            classification="infra",
            detail="enabling close requires a task lookup; none was provided.",
        )
        return receipt
    row = lookup_task(unblocks)
    if row is None:
        receipt.update(
            ok=False,
            classification="missing",
            detail=f"unblocks {unblocks} does not exist on the board.",
        )
        return receipt
    target_effect = classify_effect(row.get("production_effect"))
    receipt["unblocks_effect"] = target_effect
    if target_effect != "production":
        receipt.update(
            ok=False,
            classification="invalid",
            detail=(
                f"unblocks {unblocks} is {target_effect}, not production. "
                "An enabling card may only close against a production card."
            ),
        )
        return receipt
    receipt["classification"] = "unblocks"
    return receipt


def column_present(conn: sqlite3.Connection) -> bool:
    rows = conn.execute("PRAGMA table_info(tasks)").fetchall()
    names = {str(row[1] if not isinstance(row, sqlite3.Row) else row["name"]) for row in rows}
    return "production_effect" in names


def migrate_add_column(conn: sqlite3.Connection) -> bool:
    """ADD COLUMN if missing. Does NOT backfill. Returns True if it added."""
    if column_present(conn):
        return False
    conn.execute(f"ALTER TABLE tasks ADD COLUMN {COLUMN_DDL}")
    return True


def count_effects(
    conn: sqlite3.Connection, *, done_only: bool = False
) -> dict[str, int]:
    """Count production/enabling/internal. Missing column → all internal."""
    counts = {"production": 0, "enabling": 0, "internal": 0}
    if not column_present(conn):
        sql = "SELECT COUNT(*) FROM tasks"
        params: tuple = ()
        if done_only:
            sql += " WHERE status = 'done'"
        n = int(conn.execute(sql, params).fetchone()[0])
        counts["internal"] = n
        return counts
    sql = "SELECT production_effect, COUNT(*) FROM tasks"
    if done_only:
        sql += " WHERE status = 'done'"
    sql += " GROUP BY production_effect"
    for value, n in conn.execute(sql).fetchall():
        counts[classify_effect(value)] += int(n)
    return counts


def format_cycle_summary_line(
    all_counts: Mapping[str, int], done_counts: Mapping[str, int]
) -> str:
    """Additive CYCLE-SUMMARY line. Old `kanban: ready=...` line stays."""
    return (
        "  kanban_effect: "
        f"production={all_counts.get('production', 0)}/{done_counts.get('production', 0)} "
        f"enabling={all_counts.get('enabling', 0)}/{done_counts.get('enabling', 0)} "
        f"internal={all_counts.get('internal', 0)}/{done_counts.get('internal', 0)} "
        "(all/done)"
    )
