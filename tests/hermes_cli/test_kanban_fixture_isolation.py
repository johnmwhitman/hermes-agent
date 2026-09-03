"""Regression: fixture rows must never reach the REAL production board.

t_658c6048: a card titled "racer 0" (and siblings ``racer N``, ``ro
task``, ``Task A``, ``specimen target``) landed in the live board at
``~/.hermes/kanban.db`` while the kanban suite ran. Every one of those
titles traces to a kanban test fixture, so the suite's isolation had a
hole: a write whose resolved path pointed at the real board WITHOUT
tripping the conftest deny-list (rebuilt subprocess env, leaked
``HERMES_KANBAN_DB``, memoised root, symlinked home).

These tests pin the two new layers closed:

1. ``kanban_db.connect`` refuses, under pytest, to open ANY path that
   resolves to the real production board — no matter how the path was
   produced.
2. The autouse ``_kanban_closed_world`` fixture pins
   ``HERMES_KANBAN_DB`` to a per-test tmp file so even a subprocess that
   rebuilds its environment inherits a sandboxed board.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb


_WORKTREE = Path(__file__).resolve().parents[2]


class TestProductionBoardTrap:
    def test_connect_to_real_board_raises_under_pytest(self):
        """connect() with the real board path must refuse, not open.

        Either guard may fire first: the conftest deny-list (#69283, when the
        launch env pin was captured) or the module-level production trap
        (t_658c6048, which needs no captured env). Both are refusal."""
        real = (Path.home() / ".hermes" / "kanban.db")
        with pytest.raises(RuntimeError, match=r"production board|kanban_write_guard"):
            kb.connect(db_path=real)

    def test_connect_via_symlinked_path_raises(self, tmp_path):
        """A symlink pointing at the real board is still the real board."""
        real = (Path.home() / ".hermes" / "kanban.db")
        if not real.exists():
            pytest.skip("no production board on this host")
        link = tmp_path / "innocent-looking.db"
        link.symlink_to(real.resolve())
        with pytest.raises(RuntimeError, match=r"production board|kanban_write_guard"):
            kb.connect(db_path=link)

    def test_is_production_board_path_resolution(self):
        real = (Path.home() / ".hermes" / "kanban.db")
        assert kb._is_production_board_path(real) is True
        assert kb._is_production_board_path(Path("/tmp/definitely-not-it.db")) is False

    def test_connect_to_tmp_path_still_works(self, tmp_path):
        """The trap must not break the sandboxed path every test uses."""
        conn = kb.connect(db_path=tmp_path / "board.db")
        try:
            conn.execute("SELECT 1 FROM sqlite_master LIMIT 1")
        finally:
            conn.close()


class TestClosedWorldFixture:
    def test_argless_connect_resolves_to_per_test_home(self):
        """The per-test HERMES_HOME redirect (autouse) must be what argless
        connect() resolves against — never the real home."""
        resolved = kb.kanban_db_path()
        real_home = (Path.home() / ".hermes").resolve()
        assert not str(resolved).startswith(str(real_home)), resolved
        conn = kb.connect()
        try:
            kb.create_task(conn, title="fixture card that must stay sandboxed", assignee="dev")
        finally:
            conn.close()

    def test_subprocess_inheriting_test_env_stays_sandboxed(self, tmp_path):
        """A child that inherits the test process env (HERMES_HOME redirect +
        HERMES_TEST_ISOLATION marker) resolves to the sandboxed home."""
        child_code = (
            "import os, sys; sys.path.insert(0, %r);"
            "from hermes_cli import kanban_db as kb;"
            "from pathlib import Path;"
            "p = kb.kanban_db_path();"
            "real = (Path.home() / '.hermes').resolve();"
            "assert not str(p).startswith(str(real)), p;"
            "print(str(p))"
        ) % (str(_WORKTREE),)
        env = dict(os.environ)
        env["PYTHONPATH"] = str(_WORKTREE)
        res = subprocess.run(
            [sys.executable, "-c", child_code],
            env=env, capture_output=True, text=True, timeout=60,
        )
        assert res.returncode == 0, res.stderr

    def test_subprocess_without_pin_is_refused_by_module_trap(self, tmp_path):
        """Even if the child env loses the HERMES_HOME redirect, an explicit
        connect() at the real board must be refused by the module trap (via
        the HERMES_TEST_ISOLATION marker the conftest exports)."""
        child_code = (
            "import sys; sys.path.insert(0, %r);"
            "from hermes_cli import kanban_db as kb;"
            "from pathlib import Path;"
            "real = Path.home() / '.hermes' / 'kanban.db';"
            "raised = False\n"
            "try:\n"
            "    kb.connect(db_path=real)\n"
            "except RuntimeError as e:\n"
            "    raised = ('production board' in str(e)) or ('kanban_write_guard' in str(e))\n"
            "assert raised, 'connect to real board was NOT refused'\n"
            "print('refused-ok')"
        ) % (str(_WORKTREE),)
        env = dict(os.environ)
        env["PYTHONPATH"] = str(_WORKTREE)
        env.pop("HERMES_KANBAN_DB", None)   # strip the pin — trap must catch it
        env.pop("HERMES_HOME", None)        # strip the redirect too
        # HERMES_TEST_ISOLATION survives (conftest exports it session-wide);
        # that is by design: it is OUR marker, inherited by children.
        res = subprocess.run(
            [sys.executable, "-c", child_code],
            env=env, capture_output=True, text=True, timeout=60,
        )
        assert res.returncode == 0, res.stderr
        assert "refused-ok" in res.stdout
