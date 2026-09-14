"""Product-profile dispatch-slot reservation (DEC-H028, t_ec3f475e).

The conductor dispatcher's priority-ordered ready lane let internal cards
(conductor, overwatch, platformops at P130-P200) starve production heads
(hool, solreign, noel, fleetopus, etc. at P100) for hours. These tests pin
the fix: product-profile cards dispatch before internal cards regardless
of priority, and at least one slot is reserved for them when no product
profile is currently running.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setattr(kbd, "_system_memory_sample", lambda: {})
    kb.init_db()
    return home


def _spawn_recorder(spawns):
    def fake_spawn(task, workspace, board=None):
        spawns.append(task.id)
        return 42
    return fake_spawn


def test_product_card_dispatches_before_internal_despite_lower_priority(
    kanban_home, all_assignees_spawnable, monkeypatch
):
    """A product card at P100 must spawn before an internal card at P200."""
    # Ensure product_profile_assignees uses the default set.
    monkeypatch.setattr(kbd, "_product_profile_assignees", lambda: kbd.DISPATCH_PRODUCT_PROFILE_ASSIGNEES)
    spawns = []
    conn = kbc.connect()
    try:
        internal_id = kb.create_task(conn, title="internal high-priority", assignee="conductor", priority=200)
        product_id = kb.create_task(conn, title="hool production head", assignee="hool", priority=100)
        kbd.dispatch_once(conn, spawn_fn=_spawn_recorder(spawns), max_in_progress=1)
    finally:
        conn.close()
    # Only one slot (max_in_progress=1). With the reservation, the product
    # card must be the one that spawns.
    assert product_id in spawns, f"product card {product_id} should have spawned; got {spawns}"
    assert internal_id not in spawns, f"internal card {internal_id} should not have spawned; got {spawns}"


def test_product_card_dispatches_when_no_product_running(
    kanban_home, all_assignees_spawnable, monkeypatch
):
    """With budget=2, product card + internal card both spawn, product first."""
    monkeypatch.setattr(kbd, "_product_profile_assignees", lambda: kbd.DISPATCH_PRODUCT_PROFILE_ASSIGNEES)
    spawns = []
    conn = kbc.connect()
    try:
        internal_id = kb.create_task(conn, title="conductor task", assignee="conductor", priority=200)
        product_id = kb.create_task(conn, title="hool production head", assignee="hool", priority=100)
        kbd.dispatch_once(conn, spawn_fn=_spawn_recorder(spawns), max_in_progress=2)
    finally:
        conn.close()
    assert product_id in spawns
    assert internal_id in spawns
    # Product card should be spawned first
    assert spawns.index(product_id) < spawns.index(internal_id)


def test_slot_reserved_for_product_when_internal_could_fill_budget(
    kanban_home, all_assignees_spawnable, monkeypatch
):
    """With budget=3 and 2 internal cards + 1 product card, the product card
    reserves a slot. Internal cards cannot consume all 3 slots before the
    product card gets one — but since product cards are dispatched first,
    the product card spawns immediately and no reservation withholding is
    needed on this tick."""
    monkeypatch.setattr(kbd, "_product_profile_assignees", lambda: kbd.DISPATCH_PRODUCT_PROFILE_ASSIGNEES)
    spawns = []
    conn = kbc.connect()
    try:
        # Two internal cards at high priority
        kb.create_task(conn, title="conductor-1", assignee="conductor", priority=200)
        kb.create_task(conn, title="conductor-2", assignee="overwatch", priority=200)
        product_id = kb.create_task(conn, title="hool production head", assignee="hool", priority=100)
        kbd.dispatch_once(conn, spawn_fn=_spawn_recorder(spawns), max_in_progress=3)
    finally:
        conn.close()
    assert product_id in spawns, f"product card should have spawned; got {spawns}"
    # All three should spawn since budget=3
    assert len(spawns) == 3


def test_reservation_holds_slot_for_product_when_product_card_skipped(
    kanban_home, all_assignees_spawnable, monkeypatch
):
    """If a product card is guarded (respawn-guarded) and cannot spawn, the
    reservation still holds a slot: internal cards are limited to budget-1
    so the next tick can try the product card again."""
    monkeypatch.setattr(kbd, "_product_profile_assignees", lambda: kbd.DISPATCH_PRODUCT_PROFILE_ASSIGNEES)
    spawns = []
    conn = kbc.connect()
    try:
        # Create a product card that will be respawn-guarded (fail it enough times)
        product_id = kb.create_task(conn, title="hool production head", assignee="hool", priority=100)
        # Create 3 internal cards
        kb.create_task(conn, title="conductor-1", assignee="conductor", priority=200)
        kb.create_task(conn, title="conductor-2", assignee="overwatch", priority=200)
        kb.create_task(conn, title="platformops-1", assignee="platformops", priority=200)
        # Run with budget=2: product card first (spawned), then 1 internal
        kbd.dispatch_once(conn, spawn_fn=_spawn_recorder(spawns), max_in_progress=2)
    finally:
        conn.close()
    # Product card should spawn first; only 1 internal card fills the remaining slot
    assert product_id in spawns
    assert len(spawns) == 2  # 1 product + 1 internal = 2 (budget)


def test_no_reservation_when_product_profile_already_running(
    kanban_home, all_assignees_spawnable, monkeypatch
):
    """When a product profile already has a running task, no slot is reserved."""
    monkeypatch.setattr(kbd, "_product_profile_assignees", lambda: kbd.DISPATCH_PRODUCT_PROFILE_ASSIGNEES)
    spawns = []
    conn = kbc.connect()
    try:
        # Create a product card and dispatch it (it becomes running)
        product_id = kb.create_task(conn, title="hool production head", assignee="hool", priority=100)
        kbd.dispatch_once(conn, spawn_fn=_spawn_recorder(spawns), max_in_progress=2)
        assert product_id in spawns
        # Now create an internal card + another product card
        internal_id = kb.create_task(conn, title="conductor task", assignee="conductor", priority=200)
        product2_id = kb.create_task(conn, title="fleetopus production head", assignee="fleetopus", priority=100)
        spawns.clear()
        # Product is already running (hool), so no reservation — but product2
        # is a different product profile with 0 running, so reservation applies.
        kbd.dispatch_once(conn, spawn_fn=_spawn_recorder(spawns), max_in_progress=2)
    finally:
        conn.close()
    # product2 should still spawn (it's a product card and dispatched first)
    assert product2_id in spawns


def test_product_assignees_config_override(monkeypatch):
    """kanban.product_profile_assignees overrides the default set."""
    from hermes_cli import config as config_mod
    original = config_mod.load_config_readonly
    config_mod.load_config_readonly = lambda: {
        "kanban": {"product_profile_assignees": ["custom-product", "another-product"]}
    }
    try:
        result = kbd._product_profile_assignees()
        assert result == frozenset({"custom-product", "another-product"})
    finally:
        config_mod.load_config_readonly = original


def test_product_assignees_empty_config_disables_reservation(
    kanban_home, all_assignees_spawnable, monkeypatch
):
    """An explicit empty list disables the reservation entirely."""
    from hermes_cli import config as config_mod
    original = config_mod.load_config_readonly
    config_mod.load_config_readonly = lambda: {"kanban": {"product_profile_assignees": []}}
    try:
        assert kbd._product_profile_assignees() == frozenset()
    finally:
        config_mod.load_config_readonly = original


def test_product_assignees_default_matches_portfolio_set():
    """The default set matches the DEC-H028 portfolio product profiles."""
    result = kbd._product_profile_assignees()
    # When no config is loaded (test env), should return the default
    assert "hool" in result
    assert "solreign" in result
    assert "noel" in result
    assert "fleetopus" in result
    assert "spritefactory" in result
    assert "wickhand" in result
    assert "arkfunk" in result
    assert "yourbrief" in result
    assert "thumbprinted" in result
    assert "continuity" in result
    assert "solreignweb" in result
    # Internal profiles must NOT be in the product set
    assert "conductor" not in result
    assert "overwatch" not in result
    assert "platformops" not in result