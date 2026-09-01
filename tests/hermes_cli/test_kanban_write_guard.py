"""#69283: kanban write guard prevents tests from writing to real ~/.hermes."""

from __future__ import annotations

import pytest

from hermes_cli import kanban_db


def test_connect_succeeds_under_test_home(tmp_path, monkeypatch):
    """When HERMES_HOME is a temp dir, kanban connect succeeds normally."""
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    conn = kanban_db.connect()
    try:
        assert str(kanban_db.kanban_db_path()).startswith(str(home))
    finally:
        conn.close()


def test_connect_raises_when_kanban_home_is_real_root(monkeypatch, real_kanban_root):
    """When kanban paths resolve to the REAL root, connect raises RuntimeError."""
    monkeypatch.setattr(
        kanban_db, "kanban_home", lambda: real_kanban_root
    )
    monkeypatch.setattr(
        kanban_db,
        "kanban_db_path",
        lambda board=None: real_kanban_root / "kanban.db",
    )
    with pytest.raises(RuntimeError, match="kanban_write_guard"):
        kanban_db.connect()


def test_connect_raises_for_explicit_db_path_under_real_root(real_kanban_root):
    """Explicit db_path pointing under the real root is also refused."""
    with pytest.raises(RuntimeError, match="kanban_write_guard"):
        kanban_db.connect(real_kanban_root / "kanban.db")
