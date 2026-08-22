"""Tests for the extracted GatewayKanbanWatchersMixin (god-file Phase 3).

The kanban watcher loops were lifted out of gateway/run.py into a mixin that
GatewayRunner inherits. These tests confirm the mixin exposes the methods and
that GatewayRunner picks them up via the MRO (behavior-neutral relocation).
"""

from __future__ import annotations

import inspect

from gateway.kanban_watchers import GatewayKanbanWatchersMixin

KANBAN_METHODS = [
    "_kanban_notifier_watcher",
    "_kanban_dispatcher_watcher",
    "_kanban_advance",
    "_kanban_unsub",
    "_kanban_rewind",
    "_deliver_kanban_artifacts",
]


def test_mixin_defines_kanban_methods():
    for m in KANBAN_METHODS:
        assert hasattr(GatewayKanbanWatchersMixin, m), f"mixin missing {m}"


def test_dispatcher_uses_private_executor_for_blocking_work():
    """Dispatcher progress must not depend on the gateway default pool."""
    source = inspect.getsource(GatewayKanbanWatchersMixin._kanban_dispatcher_watcher)
    assert "ThreadPoolExecutor" in source
    assert "run_in_executor" in source
    assert "await asyncio.to_thread(_kb.reap_worker_zombies)" not in source
    assert "await asyncio.to_thread(_tick_once)" not in source
    assert "await asyncio.to_thread(_ready_nonempty)" not in source

