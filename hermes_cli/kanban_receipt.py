"""Hermes Kanban result contract — the ``hermes.kanban-result/v1`` envelope.

Card t_2d7fd66c (R1-Hermes sub-deliverable from the Kanban receipt dogfood
design, sha256 ``b938cb740122685e349d56ea3ebd45687d1a0bdb9deef4d9330d754e9e1275bb``).

WHAT THIS FILE OWNS
-------------------
1. The **result-contract normalizer** — turns a closing worker's
   ``(summary, result, metadata, artifacts)`` into the canonical v1 envelope
   that MeshFleet's ``record_work_receipt`` accepts. The byte layout must
   match ``computeWorkReceiptPayloadSha256`` in
   ``agent-mesh/src/work-receipt.ts`` exactly, including the
   ``JSON.stringify`` key order and the ``(kind, handle, digest)``
   ascending sort on the evidence array. Any drift here is a wire-format
   conflict at the MCP boundary and a ``payload_sha256`` rejection.

2. The **canonical evidence gate** — a single function that maps the
   per-card prose evidence (``existing_paths``, ``existing_shas``,
   ``attachments``, ``has_verified``+``has_cmd``) into the v1
   ``evidence[]`` array. The classifier is the one already living in
   ``tools/kanban_tools.py`` (``_receipt_classify``); we re-import it so
   the gate's verdict is consistent across the tool, the CLI, the
   dashboard, and the swarm path.

3. The **strict numerus / denominator mapping** — turning the gate's
   outcome into the v1 enum triple ``(terminal_outcome, result_contract,
   quality_gate)``. The combinator table is the one the MeshFleet
   validator enforces on the other side; replicating it here keeps
   ``validateWorkReceipt`` from refusing every row we mint.

WHAT THIS FILE DOES NOT OWN
---------------------------
* The outbox / reconciler — that is ``hermes_cli.kanban_receipt_outbox``
  and ``hermes_cli.kanban_receipt_reconciler`` (siblings). Keeping the
  normalizer separate means the wire-format unit tests do not have to
  fake a SQLite database or a MeshFleet process.

* The schema migration — that lives in ``hermes_cli/kanban_db.py`` next
  to the rest of the additive migrations, so a legacy board opens
  cleanly.

* The transport call — ``kanban_db.complete_task`` writes the outbox row
  in the same write_txn as the status flip; the reconciler is the only
  caller of the MCP ``record_work_receipt`` tool, and it is a process
  that runs out-of-band (cron / launchd-supervised worker). The card
  close itself never depends on MeshFleet being reachable — that is the
  *fail-open* clause in design §6 (fail-open card, fail-closed metric
  credit).

DESIGN-LEVEL INVARIANTS (RATIFIED, SHA ``b938cb...`` §4-§7)
------------------------------------------------------------
* Idempotency key: ``(source='hermes-kanban', task_id, run_id)``. A
  replay of byte-equivalent bytes returns the existing row; a replay of
  the same key with different bytes is refused and never overwrites
  history.

* Strict numerus (R1 §4.2): the production-quality numerator includes
  only rows where ``terminal_outcome == 'completed' AND result_contract
  in {'ok', 'artifact_missing'} AND quality_gate == 'passed' AND
  evidence.length >= 1``. Anything else is in the denominator only.

* Evidence minimization (R1 §4.4): never include the card body, raw
  model output, raw secrets, or absolute private paths in
  ``evidence[]``. We hash paths on disk (attachment + artifact kind)
  and reference git SHAs as the canonical handle for ``git_commit``
  kind; ``command_run`` carries the first argv-shaped fragment and a
  pre-computed digest when one is available.

* Determinism: the same inputs must produce the same
  ``payload_sha256`` byte-for-byte, regardless of platform or run order.
  The serializer below writes the same explicit object-key order as the
  JavaScript ``JSON.stringify`` implementation and sorts evidence with
  JavaScript UTF-16 code-unit comparison.

This module is pure — no I/O, no time, no ``time.time()``. The caller
is responsible for ``completed_at`` and ``recorded_at``; we return them
as inputs to the v1 envelope and never read a clock.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# Wire constants — must match the MeshFleet ``work-receipt.ts`` schema marker
# and the verifier v3 ``work_receipt.*`` checks.
# ---------------------------------------------------------------------------

WORK_RECEIPT_SCHEMA = "hermes.kanban-result/v1"
"""The single schema marker. Bumping this is wire-incompatible."""

WORK_RECEIPT_SOURCE = "hermes-kanban"
"""The single source string the Hermes contract always uses."""

# Stable enum strings — kept as separate frozensets so the gate can
# detect "impossible" combinations locally before the wire. The MeshFleet
# validator re-checks them, but the local check gives us a faster
# failure path and a useful error message.
TERMINAL_OUTCOMES = frozenset({"completed", "failed", "refused", "blocked"})
RESULT_CONTRACT_STATUSES = frozenset(
    {"ok", "refused", "blocked", "artifact_missing", "invalid", "absent"}
)
QUALITY_GATE_STATUSES = frozenset({"passed", "failed"})
EVIDENCE_KINDS = frozenset(
    {"git_commit", "attachment", "artifact", "command_run", "external"}
)

# Grammar checks lifted from work-receipt.ts. The MeshFleet side re-checks
# them; mirroring here gives a clean local ``ok=False, reasons=[...]``
# before we touch the database.
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
TASK_ID_GRAMMAR = re.compile(r"^t_[A-Za-z0-9]+$")
EVIDENCE_DIGEST_GRAMMAR = HEX_SHA256  # alias for symmetry

#: Minimum number of chars required for a CLI/CLI-attached ``command_run``
#: handle. Empty handles round-trip as ``null`` (omitted) on the wire and
#: are never written as evidence.
_COMMAND_RUN_MIN_HANDLE_CHARS = 1

#: The maximum number of evidence entries we will record per receipt.
#: A worker can declare many; the strict numerus only needs ``>=1`` but
#: writing the full set is honest and audit-friendly.
_MAX_EVIDENCE_ENTRIES = 64

#: The maximum number of artifact paths the normalizer accepts. Beyond
#: this we drop the tail with a warning event (logged upstream) — see
#: ``_truncate_evidence`` for the truncation contract.
_MAX_ARTIFACTS = 32


# ---------------------------------------------------------------------------
# Public dataclass-equivalents (plain dicts are fine; we don't want a
# dataclass dependency in this hot path). The shape is documented here so
# callers know what to expect.
# ---------------------------------------------------------------------------


class ResultContractError(ValueError):
    """Raised when inputs cannot be turned into a valid v1 envelope.

    Carries a list of human-readable reasons so the gate can surface
    them in ``task_events`` and the dashboard can render them as a
    bullet list. The MeshFleet validator (``validateWorkReceipt``) is
    the final authority; this exception is only used locally to fail
    closed before the outbox row is written.
    """

    def __init__(self, reasons: list[str]) -> None:
        if not reasons:
            reasons = ["unknown rejection"]
        self.reasons = list(reasons)
        super().__init__("; ".join(reasons))


# ---------------------------------------------------------------------------
# Canonical-evidence gate — re-imports the existing classifier so the
# tool, CLI, dashboard, and swarm path agree on what counts as
# "observable receipt" prose. Tests in tests/hermes_cli/ already cover
# the classifier; we do not duplicate that surface here.
# ---------------------------------------------------------------------------

# Deferred import — ``tools.kanban_tools`` is a runtime entry, not part
# of the hermes_cli package proper. The import is cheap and only
# happens on first call to ``classify_prose_receipt``.
def _import_receipt_classifier():
    from tools.kanban_tools import _receipt_classify  # type: ignore

    return _receipt_classify


def _import_receipt_repo_helpers():
    """Lazy import for the multi-repo resolver helpers added in t_cca67626.

    Keeps ``hermes_cli.kanban_receipt`` importable even when
    ``tools.kanban_tools`` is partially installed or has a syntax error
    — the gate must never break the kanban CLI on import. The helpers
    are exported from the tools module to share the multi-repo logic
    with the tool layer (gate, dashboard verifier).
    """
    try:
        from tools.kanban_tools import (  # type: ignore
            _receipt_repo_candidates,
            _RECEIPT_AI_ROOT,
        )

        return _receipt_repo_candidates, _RECEIPT_AI_ROOT
    except Exception:
        return None, None


def _task_repo_candidates(conn: Any, task_id: str) -> list[Path] | None:
    """Resolve the ordered list of git repos the SHA resolver should
    consult for ``task_id``.

    Reads the ``workspace_path`` (and ``project_id``, when set) from
    the task row, then defers to ``tools.kanban_tools.
    _receipt_repo_candidates`` for the actual resolution. Returns
    ``None`` when the helpers cannot be imported — callers must
    treat that as "no candidate list supplied" and fall back to the
    historical single-root default.

    Pure helper: never mutates ``conn``. Tolerates any failure (table
    missing, projects_db missing, path absent on disk) — a missing
    candidate is simply absent from the returned list, which is
    exactly what the resolver wants.
    """
    _repo_candidates_fn, _ai_root = _import_receipt_repo_helpers()
    if _repo_candidates_fn is None:
        return None

    workspace_path: str | None = None
    project_id: str | None = None
    try:
        row = conn.execute(
            "SELECT workspace_path, project_id FROM tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
    except Exception:
        row = None
    if row is not None:
        try:
            workspace_path = row["workspace_path"] if "workspace_path" in row.keys() else None
        except (AttributeError, KeyError):
            workspace_path = None
        try:
            project_id = row["project_id"] if "project_id" in row.keys() else None
        except (AttributeError, KeyError):
            project_id = None

    project_repo: str | None = None
    if project_id:
        # Defer projects_db import — it lives in hermes_cli and is
        # always importable from this module. A missing project row
        # is not an error: it just means no project repo candidate.
        try:
            from hermes_cli import projects_db as _pdb  # type: ignore

            try:
                with _pdb.connect_closing() as _pconn:
                    project_obj = _pdb.get_project(_pconn, project_id)
            except Exception:
                project_obj = None
            if project_obj is not None:
                # Project is a dataclass / object — try the documented
                # attribute first, then fall back to a generic attr
                # lookup. Both ProjectsDB versions live on the same
                # shape (``primary_repo_path``), but we tolerate either.
                project_repo = getattr(project_obj, "primary_repo_path", None) or getattr(
                    project_obj, "primary_repo", None
                )
        except Exception:
            project_repo = None

    try:
        return _repo_candidates_fn(
            workspace_path,
            project_repo,
            fallback=_ai_root,
        )
    except Exception:
        return None


def _classify_prose(
    conn: Any,
    task_id: str,
    summary: str,
    result: str,
    artifacts: Iterable[str],
    *,
    repos: list[Path] | None = None,
) -> dict[str, Any]:
    """Run the existing ``_receipt_classify`` over the card's prose.

    Returns the same dict shape the tool layer returns, plus the
    ``attachments`` list (already filtered to existing attachment ids).
    ``conn`` is the kanban SQLite handle so the classifier can resolve
    existing attachment paths.

    ``repos`` is forwarded to ``_receipt_classify`` as the multi-repo
    SHA candidate list. When omitted, the function computes the
    candidate list from the task row (workspace_path → project_id →
    ``~/AI`` fallback). Callers that already have a candidate list may
    pass it through to avoid an extra DB round-trip.
    """
    receipt_classify = _import_receipt_classifier()
    from tools.kanban_tools import _receipt_existing_attachments  # type: ignore

    attachments = _receipt_existing_attachments(conn, task_id)
    receipt_text_parts = [summary or "", result or ""]
    receipt_text_parts.extend(str(p) for p in artifacts if p)
    receipt_text = "\n".join(part for part in receipt_text_parts if part)
    if repos is None:
        repos = _task_repo_candidates(conn, task_id)
    return receipt_classify(receipt_text, attachments, repos=repos)


# ---------------------------------------------------------------------------
# Evidence-array normalizer. The MeshFleet side validates each entry
# (``kind`` is in the enum, ``handle`` is non-empty, ``digest`` if
# present matches ``^[0-9a-f]{64}$``) and requires the array to be
# sorted by ``(kind, handle, digest)`` ascending before the SHA-256 is
# computed. We do the sort here so the wire-side validator can do
# straight equality on the byte payload.
# ---------------------------------------------------------------------------


def _hash_file_sha256(path: str) -> str | None:
    """Return the SHA-256 of ``path`` as 64-char hex, or ``None`` if the
    file is unreadable / absent / too large to safely read.

    Used for the ``artifact`` evidence kind — MeshFleet records the
    digest but does not fetch the file, so a local computation is the
    only source of truth for that handle. The cap is 256 MiB so a runaway
    very-large attachment does not stall the close-time write.
    """
    import os

    if not path:
        return None
    try:
        st = os.stat(path)
    except OSError:
        return None
    if not (st.st_mode & 0o170000 == 0o100000):  # regular file only
        return None
    if st.st_size > 256 * 1024 * 1024:
        return None
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fp:
            for chunk in iter(lambda: fp.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _resolve_evidence(
    conn: Any,
    task_id: str,
    *,
    summary: str,
    result: str,
    metadata: dict[str, Any] | None,
    artifacts: list[str],
) -> list[dict[str, Any]]:
    """Translate the card's evidence into the canonical v1 evidence list.

    The normalization rules below are LOAD-BEARING for the contract;
    the operator's mid-run review (codex-root, attached to t_2d7fd66c)
    flagged three production-level bugs that this function used to
    ship. Every rule here defends against one of them:

    * ``git_commit`` kind is only emitted for a SHA that the
      operator's existing-shas resolver (git cat-file -e) accepts — the
      same check the ``_receipt_classify`` tool-layer helper runs. A
      bare hex string in the prose is NOT evidence; admission requires
      the resolver's green light.

    * ``attachment`` kind carries an **opaque, content-derived
      handle** (``att:<sha256>``). The raw stored_path is filesystem
      state and may include absolute private paths under
      ``/Users/johnwhitman/...``; the design's minimization boundary
      (R1 §4.4) forbids serializing those into a cross-process wire
      payload. The digest is the SHA-256 of the file bytes — also
      MeshFleet's content-addressed identity, so an equal-digest
      handle is still a stable replay key.

    * ``artifact`` kind is the same pattern: an **opaque handle**
      (``art:<sha256>``) backed by the SHA-256 of the on-disk bytes.
      The absolute path is NEVER placed on the wire; the adapter
      that needs to re-resolve it keeps the path table private.

    * ``command_run`` kind is NEVER manufactured from the prose. The
      operator's invariant is explicit: a hash of a self-attested
      VERIFIED line is not execution evidence. The only path that
      emits ``command_run`` is a caller-supplied
      ``metadata.external_evidence`` entry (kind=command_run) backed
      by a separately-stored record — typically written by the
      command-execution harness. Prose claiming ``VERIFIED bash …``
      earns ``quality_gate=passed`` (the prose path is a gate, not an
      evidence kind) but is NEVER itself serialized as evidence.

    * ``external`` kind is the only non-attested-by-hermes evidence;
      it is recorded verbatim from ``metadata.external_evidence``.

    The function is intentionally permissive: it returns whatever
    evidence it can find. The strict numerus check happens in
    :func:`build_envelope`.
    """
    from tools.kanban_tools import _receipt_existing_attachments  # type: ignore

    evidence: list[dict[str, Any]] = []

    # ---- Prose-side classification: the operator's classifier is the
    # ---- authoritative source for what counts as observable receipt.
    receipt_prose = _classify_prose(
        conn,
        task_id,
        summary or "",
        result or "",
        artifacts,
    )

    # ---- git_commit (only existing, repo-resolvable SHAs) ----------
    for sha in receipt_prose.get("existing_shas") or []:
        evidence.append({"kind": "git_commit", "handle": sha})

    # ---- attachments (opaque handles, private paths never on wire) -
    attachments = _receipt_existing_attachments(conn, task_id)
    for att in attachments:
        # Normalize the lookup so we accept both stored-path strings
        # and dict-shaped rows.
        if isinstance(att, dict):
            path = att.get("path") or att.get("stored_path") or ""
        else:
            path = str(att)
        if not path:
            continue
        digest = _hash_file_sha256(path)
        # Fall back to a digest the row may already carry (uploaded
        # under a non-readable mode). Only honor hex-64 strings.
        if digest is None:
            candidate = (
                att.get("sha256") if isinstance(att, dict) else None
            )
            if (
                isinstance(candidate, str)
                and HEX_SHA256.match(candidate)
            ):
                digest = candidate
        if digest is None:
            # Path unreadable AND no precomputed digest: drop. The
            # upstream classifier already gated the close on this.
            continue
        evidence.append(
            {
                "kind": "attachment",
                "handle": _opaque_handle("att", digest),
                "digest": digest,
            }
        )

    # ---- artifacts (opaque handles, private paths never on wire) ---
    seen_digests: set[str] = {e["digest"] for e in evidence if "digest" in e}
    for path in artifacts or []:
        if not isinstance(path, str) or not path.strip():
            continue
        digest = _hash_file_sha256(path)
        if digest is None:
            # Path was claimed but unreadable; drop here.
            continue
        if digest in seen_digests:
            continue
        seen_digests.add(digest)
        evidence.append(
            {
                "kind": "artifact",
                "handle": _opaque_handle("art", digest),
                "digest": digest,
            }
        )

    # ---- classifier existing_paths (C20 / t_2531481b) ---------------
    # ``_derive_triple`` used to treat ``existing_paths`` as evidence
    # (quality_gate=passed) while this function never serialized them.
    # Honest VERIFIED completions that cite an on-disk file — the live
    # ``receipt_outbox_rejected`` class after t_02d3dba9 — then failed
    # the MeshFleet cross-field check. Hash each existing regular file
    # into the same opaque ``artifact`` shape as caller-supplied
    # artifacts; directories / unreadable paths stay dropped.
    for raw in receipt_prose.get("existing_paths") or []:
        if not isinstance(raw, str) or not raw.strip():
            continue
        digest = _hash_file_sha256(os.path.expanduser(raw.strip()))
        if digest is None or digest in seen_digests:
            continue
        seen_digests.add(digest)
        evidence.append(
            {
                "kind": "artifact",
                "handle": _opaque_handle("art", digest),
                "digest": digest,
            }
        )

    # ---- command_run: ONLY caller-supplied records, never prose. ---
    # The operator invariant: a hash of a self-attested VERIFIED line
    # is not execution evidence. If the worker wants a command_run
    # record on the wire, it must come from the command-execution
    # harness via ``metadata.external_evidence``. The prose path can
    # admit the *card* close (fail-open); it is never itself a wire
    # handle, and it must not stamp quality_gate=passed without a
    # serialized evidence entry (C20).

    # ---- external_evidence from metadata (incl. command_run) ------
    if isinstance(metadata, dict):
        for entry in metadata.get("external_evidence") or []:
            if not isinstance(entry, dict):
                continue
            kind = entry.get("kind")
            handle = entry.get("handle")
            digest = entry.get("digest")
            if kind not in EVIDENCE_KINDS or not isinstance(handle, str):
                continue
            if not handle.strip():
                continue
            # Belt-and-braces: the wire envelope must never carry
            # absolute private paths. If the caller passed one in,
            # drop the entry.
            if _looks_like_private_path(handle):
                continue
            normalized: dict[str, Any] = {"kind": kind, "handle": handle}
            if digest is not None:
                if not (
                    isinstance(digest, str)
                    and EVIDENCE_DIGEST_GRAMMAR.match(digest)
                ):
                    continue
                normalized["digest"] = digest
            evidence.append(normalized)

    # Trim to the per-receipt cap.
    if len(evidence) > _MAX_EVIDENCE_ENTRIES:
        evidence = evidence[:_MAX_EVIDENCE_ENTRIES]

    return evidence


def _opaque_handle(prefix: str, digest: str) -> str:
    """Return a content-derived handle that does not leak filesystem state.

    The shape is ``<prefix>:<sha256>``. The prefix disambiguates the
    kind so two equal-digest attachment + artifact entries are still
    distinguishable on the wire. The digest itself is the same
    SHA-256 MeshFleet already records in the ``digest`` field, so
    equal-digest rows are stable replay keys.

    Path-bearing handles (``/Users/john/...``) are the FAILURE case
    the operator flagged; this function never produces one.
    """
    return f"{prefix}:{digest}"


def _looks_like_private_path(handle: str) -> bool:
    """Heuristic: does this string look like an absolute private path?

    The wire-side check is stricter (it scans for ``/Users/``, ``~``,
    ``..``, UNC roots, and absolute-Unix paths); this local check is a
    safety net for callers that pass metadata-supplied handles. False
    positives are tolerable — false negatives (letting an absolute
    path through) are not.
    """
    if not isinstance(handle, str) or not handle:
        return False
    if handle.startswith("/") or handle.startswith("~"):
        return True
    if handle.startswith("\\\\"):
        return True  # UNC
    if "/" in handle and (
        handle.startswith("/") or handle.startswith("./")
    ):
        return True
    return False


def _js_key(value: str) -> tuple[int, ...]:
    """Return a JS UTF-16-code-unit compatible lexical key."""
    raw = value.encode("utf-16-be", "surrogatepass")
    return tuple(raw[i] * 256 + raw[i + 1] for i in range(0, len(raw), 2))


def _sort_evidence(
    evidence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Sort evidence by ``(kind, handle, digest)`` ascending.

    Mirrors ``computeWorkReceiptPayloadSha256`` in
    ``agent-mesh/src/work-receipt.ts`` byte-for-byte. Two equivalent
    sets presented in different orders must produce the same digest.
    """
    projected: list[dict[str, Any]] = []
    for entry in evidence:
        # JavaScript constructs a fresh object here. Reusing the caller's dict
        # would make its insertion order (and any unrelated keys) part of the
        # digest even though MeshFleet never does that.
        normalized = {
            "kind": entry["kind"],
            "handle": entry["handle"],
        }
        if "digest" in entry:
            normalized["digest"] = entry["digest"]
        projected.append(normalized)
    return sorted(
        projected,
        key=lambda e: (
            _js_key(e["kind"]),
            _js_key(e["handle"]),
            _js_key(e.get("digest") or ""),
        ),
    )


# ---------------------------------------------------------------------------
# payload_sha256 — matches ``computeWorkReceiptPayloadSha256``.
# ---------------------------------------------------------------------------


def canonical_payload(envelope: dict[str, Any]) -> dict[str, Any]:
    """Return MeshFleet's canonical caller-owned payload projection.

    ``source``, ``payload_sha256``, and ``recorded_at`` are deliberately not
    members: the first and last are server-owned and the digest is computed
    over the remaining fields. A fresh evidence object is produced for every
    item so arbitrary caller key/insertion order cannot affect bytes.
    """
    return {
        "schema": envelope["schema"],
        "task_id": envelope["task_id"],
        "run_id": envelope["run_id"],
        "assignee": envelope["assignee"],
        "terminal_outcome": envelope["terminal_outcome"],
        "result_contract": envelope["result_contract"],
        "quality_gate": envelope["quality_gate"],
        "completed_at": envelope["completed_at"],
        "evidence": _sort_evidence(envelope["evidence"]),
    }


def compute_payload_sha256(envelope: dict[str, Any]) -> str:
    """Return the canonical SHA-256 of the v1 envelope.

    Field order, JSON key order, and evidence-array ordering all matter.
    The canonical object uses the explicit insertion order accepted by
    MeshFleet's JSON.stringify implementation. Evidence entries are sorted
    separately; object-key sorting would produce a different digest.
    """
    canonical = json.dumps(
        canonical_payload(envelope),
        sort_keys=False,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Triple mapping — the local equivalent of MeshFleet's cross-field
# invariants. We run the same checks so the wire validator never sees a
# locally-bad envelope.
# ---------------------------------------------------------------------------


def _derive_triple(
    *,
    task_status: str,
    receipt_prose: dict[str, Any],
    evidence_count: int = 0,
) -> tuple[str, str, str]:
    """Map the kanban close + serialized evidence onto the v1 triple.

    Returns ``(terminal_outcome, result_contract, quality_gate)`` chosen
    so MeshFleet's cross-field invariants pass. The contract:

    * ``terminal_outcome=completed`` requires ``result_contract`` in
      ``{ok, artifact_missing}`` and ``quality_gate`` ∈ ``{passed,
      failed}``. The normalizer only emits ``ok`` here, but
      ``artifact_missing`` is reserved for the future path where the
      worker declares artifacts the disk verifier cannot find.

    * ``terminal_outcome=failed|refused|blocked`` forbids
      ``result_contract=ok`` (a refused run is not ok).

    * ``quality_gate=passed`` requires ``result_contract=ok`` and at
      least one *serialized* evidence entry — the strict numerator.
      Prose-only signals (``has_verified``+``has_cmd``, or
      ``existing_paths`` that did not hash) admit the card close but
      must not stamp ``passed`` with an empty ``evidence[]`` (C20:
      that combination raises ``ResultContractError`` and writes
      ``receipt_outbox_rejected`` instead of an outbox row).

    ``receipt_prose`` is kept for callers / tests; the pass/fail bit
    is ``evidence_count``, which is the length of the array
    :func:`_resolve_evidence` actually emits.
    """
    has_evidence = evidence_count >= 1
    if task_status == "done":
        # Done is what the *card* says. The receipt's terminal_outcome
        # is what the *run* was — and the MeshFleet validator refuses
        # ``terminal_outcome=completed`` unless ``result_contract`` is
        # ``ok`` or ``artifact_missing`` (see work-receipt.ts).
        #
        # A done card with no serialized evidence therefore produces a
        # receipt that records ``terminal_outcome=failed`` and
        # ``result_contract=absent``: the card still closes (fail-open,
        # design §6), but the receipt refuses to claim a successful run.
        # Downstream numerators treat that as a denominator row, never
        # as a success. This is what the design calls "legacy cards
        # may complete but emit quality_gate=failed / result_contract=
        # absent" — the run failed its own observable-receipt gate.
        if has_evidence:
            return ("completed", "ok", "passed")
        return ("failed", "absent", "failed")
    if task_status == "blocked":
        return ("blocked", "blocked", "failed")
    # refused / failed / any other terminal (shouldn't normally happen
    # via the kanban CLI but the contract still has to be consistent)
    return ("failed", "invalid", "failed")


# ---------------------------------------------------------------------------
# Public surface — ``build_envelope`` is the single entry point used by
# ``kanban_db.complete_task``.
# ---------------------------------------------------------------------------


def build_envelope(
    conn: Any,
    task_id: str,
    *,
    run_id: int,
    assignee: str,
    completed_at: int,
    summary: str | None,
    result: str | None,
    metadata: dict[str, Any] | None,
    artifacts: list[str] | None,
    task_status: str,
) -> dict[str, Any]:
    """Build the canonical ``hermes.kanban-result/v1`` envelope.

    Returns a dict with the wire-shaped fields, including
    ``payload_sha256``. Raises :class:`ResultContractError` if the
    inputs cannot produce a valid envelope (empty assignee, malformed
    task_id, non-positive run_id, etc.).

    The function is pure modulo the SQLite ``conn`` — it reads the
    attachments table but does not write anywhere. That keeps the
    close-time atomic transaction tight: one ``INSERT INTO
    meshfleet_receipt_outbox`` row plus the existing status flip is all
    the kanban close has to do.
    """
    metadata = metadata or {}
    artifacts = list(artifacts or [])

    reasons: list[str] = []
    if not TASK_ID_GRAMMAR.match(task_id or ""):
        reasons.append(
            f"task_id must match /{TASK_ID_GRAMMAR.pattern}/, "
            f"got {task_id!r}"
        )
    if not isinstance(run_id, int) or isinstance(run_id, bool) or run_id <= 0:
        reasons.append(
            f"run_id must be a positive integer, got {run_id!r}"
        )
    # C20 class 2: system-created probe cards (kanban-rtt-probe-*) store
    # assignee NULL. Coalesce here so a caller that forgets the enqueue
    # ``or "system"`` cannot write receipt_outbox_rejected.
    if not isinstance(assignee, str) or not assignee.strip():
        assignee = "system"
    if (
        not isinstance(completed_at, int)
        or isinstance(completed_at, bool)
        or completed_at <= 0
    ):
        reasons.append(
            f"completed_at must be a positive unix-second integer, "
            f"got {completed_at!r}"
        )
    if reasons:
        raise ResultContractError(reasons)

    receipt_prose = _classify_prose(
        conn,
        task_id,
        summary or "",
        result or "",
        artifacts,
    )
    evidence = _resolve_evidence(
        conn,
        task_id,
        summary=summary or "",
        result=result or "",
        metadata=metadata,
        artifacts=artifacts,
    )
    terminal_outcome, result_contract, quality_gate = _derive_triple(
        task_status=task_status,
        receipt_prose=receipt_prose,
        evidence_count=len(evidence),
    )

    envelope: dict[str, Any] = {
        "schema": WORK_RECEIPT_SCHEMA,
        "task_id": task_id,
        "run_id": run_id,
        "assignee": assignee.strip(),
        "terminal_outcome": terminal_outcome,
        "result_contract": result_contract,
        "quality_gate": quality_gate,
        "completed_at": completed_at,
        "evidence": _sort_evidence(evidence),
    }
    envelope["payload_sha256"] = compute_payload_sha256(envelope)

    # Cross-field invariants (mirror ``validateWorkReceipt`` in
    # MeshFleet's work-receipt.ts). The local check gives us a useful
    # error message before the row ever leaves the SQLite outbox; the
    # MeshFleet side will re-check, but a failure here is almost
    # always a hermes bug worth surfacing immediately.
    cross_reasons: list[str] = []
    if (
        quality_gate == "passed"
        and result_contract != "ok"
    ):
        cross_reasons.append(
            f"quality_gate=passed requires result_contract=ok, got "
            f"result_contract={result_contract!r}"
        )
    if (
        terminal_outcome == "completed"
        and result_contract not in {"ok", "artifact_missing"}
    ):
        cross_reasons.append(
            f"terminal_outcome=completed requires result_contract in "
            f"{{ok, artifact_missing}}, got "
            f"result_contract={result_contract!r}"
        )
    if (
        terminal_outcome in {"refused", "failed"}
        and result_contract == "ok"
    ):
        cross_reasons.append(
            f"terminal_outcome={terminal_outcome} forbids "
            f"result_contract=ok"
        )
    if quality_gate == "passed" and len(evidence) == 0:
        cross_reasons.append(
            "quality_gate=passed requires at least one evidence entry"
        )
    if cross_reasons:
        raise ResultContractError(cross_reasons)

    return envelope


# ---------------------------------------------------------------------------
# Validation mirror — used by tests and the reconciler before it POSTs
# to MeshFleet. The MeshFleet side is the final authority; this is a
# cheap local check that catches drift between the schema constant
# above and the wire schema before we touch the network.
# ---------------------------------------------------------------------------


def validate_envelope(envelope: dict[str, Any]) -> list[str]:
    """Return an empty list on success, a list of reasons on failure.

    The local equivalent of ``validateWorkReceipt``. The MeshFleet side
    is the canonical check; this exists for two reasons:

    1. Unit tests can validate the normalizer's output without a
       running MeshFleet process.
    2. The reconciler can refuse a row before it leaves the SQLite
       outbox — if the schema constant here has drifted from the wire
       schema, every row would otherwise be rejected server-side and
       pollute the dead-letter table with local-bug noise.
    """
    reasons: list[str] = []
    if not isinstance(envelope, dict):
        return ["envelope is not a JSON object"]
    if envelope.get("schema") != WORK_RECEIPT_SCHEMA:
        reasons.append(
            f"schema must be the literal {WORK_RECEIPT_SCHEMA!r}, "
            f"got {envelope.get('schema')!r}"
        )
    if not TASK_ID_GRAMMAR.match(envelope.get("task_id") or ""):
        reasons.append(
            f"task_id must match /{TASK_ID_GRAMMAR.pattern}/, got "
            f"{envelope.get('task_id')!r}"
        )
    if (
        not isinstance(envelope.get("run_id"), int)
        or isinstance(envelope.get("run_id"), bool)
        or envelope["run_id"] <= 0
    ):
        reasons.append(
            f"run_id must be a positive integer, got "
            f"{envelope.get('run_id')!r}"
        )
    if not isinstance(envelope.get("assignee"), str) or not envelope["assignee"].strip():
        reasons.append(
            f"assignee must be a non-empty string, got "
            f"{envelope.get('assignee')!r}"
        )
    if envelope.get("terminal_outcome") not in TERMINAL_OUTCOMES:
        reasons.append(
            f"terminal_outcome must be one of "
            f"{sorted(TERMINAL_OUTCOMES)}, got "
            f"{envelope.get('terminal_outcome')!r}"
        )
    if envelope.get("result_contract") not in RESULT_CONTRACT_STATUSES:
        reasons.append(
            f"result_contract must be one of "
            f"{sorted(RESULT_CONTRACT_STATUSES)}, got "
            f"{envelope.get('result_contract')!r}"
        )
    if envelope.get("quality_gate") not in QUALITY_GATE_STATUSES:
        reasons.append(
            f"quality_gate must be one of "
            f"{sorted(QUALITY_GATE_STATUSES)}, got "
            f"{envelope.get('quality_gate')!r}"
        )
    if (
        not isinstance(envelope.get("completed_at"), int)
        or isinstance(envelope.get("completed_at"), bool)
        or envelope["completed_at"] <= 0
    ):
        reasons.append(
            f"completed_at must be a positive unix-second integer, "
            f"got {envelope.get('completed_at')!r}"
        )
    evidence = envelope.get("evidence")
    if not isinstance(evidence, list):
        reasons.append(
            f"evidence must be a list, got {type(evidence).__name__}"
        )
    elif len(evidence) == 0 and envelope.get("quality_gate") == "passed":
        # The strict-numerus cross-field invariant: a passed quality
        # gate with zero evidence entries would be a contradiction —
        # the wire validator refuses it; mirror the check here so
        # callers learn without a round-trip.
        reasons.append(
            "quality_gate=passed requires at least one evidence entry"
        )
    else:
        for idx, entry in enumerate(evidence):
            if not isinstance(entry, dict):
                reasons.append(
                    f"evidence[{idx}] must be a JSON object"
                )
                continue
            if entry.get("kind") not in EVIDENCE_KINDS:
                reasons.append(
                    f"evidence[{idx}].kind must be one of "
                    f"{sorted(EVIDENCE_KINDS)}, got "
                    f"{entry.get('kind')!r}"
                )
            handle = entry.get("handle")
            if not isinstance(handle, str) or not handle.strip():
                reasons.append(
                    f"evidence[{idx}].handle must be a non-empty "
                    f"string, got {handle!r}"
                )
            elif _looks_like_private_path(handle):
                # Belt-and-braces: refuse absolute private paths in
                # evidence handles regardless of where they came from.
                # The wire envelope is cross-process; leaking
                # ``/Users/...`` paths would be a minimization boundary
                # violation (R1 §4.4). The MeshFleet side does not
                # re-check this — the local check is the only one.
                reasons.append(
                    f"evidence[{idx}].handle carries a private "
                    f"absolute path, got {handle!r}"
                )
            if "digest" in entry:
                digest = entry["digest"]
                # Belt-and-braces: ``HEX_SHA256.match`` only accepts
                # str; a non-str digest raises TypeError before the
                # test for emptiness can be skipped. Catch it here so
                # callers learn the wire shape without a hard
                # exception bubbling up to the gate.
                if not isinstance(digest, str):
                    reasons.append(
                        f"evidence[{idx}].digest must be a string, "
                        f"got {type(digest).__name__}"
                    )
                elif not HEX_SHA256.match(digest):
                    reasons.append(
                        f"evidence[{idx}].digest must match "
                        f"/{HEX_SHA256.pattern}/, got {digest!r}"
                    )

    # Cross-field invariants — mirroring MeshFleet's ``validateWorkReceipt``.
    # The validator refuses to mint a row whose triple contradicts the
    # wire-side numerus; if these are absent the dead-letter table fills
    # up with locally-bug noise after every close.
    result_contract = envelope.get("result_contract")
    terminal_outcome = envelope.get("terminal_outcome")
    quality_gate = envelope.get("quality_gate")
    if (
        terminal_outcome == "completed"
        and result_contract not in {"ok", "artifact_missing"}
    ):
        reasons.append(
            f"terminal_outcome=completed requires result_contract in "
            f"{{ok, artifact_missing}}, got "
            f"result_contract={result_contract!r}"
        )
    if (
        terminal_outcome in {"refused", "failed"}
        and result_contract == "ok"
    ):
        reasons.append(
            f"terminal_outcome={terminal_outcome} forbids "
            f"result_contract=ok"
        )
    if quality_gate == "passed" and result_contract != "ok":
        reasons.append(
            f"quality_gate=passed requires result_contract=ok, got "
            f"result_contract={result_contract!r}"
        )

    if not reasons and envelope.get("payload_sha256") != compute_payload_sha256(envelope):
        reasons.append(
            "payload_sha256 does not match the canonical digest of the "
            "supplied fields"
        )

    return reasons
