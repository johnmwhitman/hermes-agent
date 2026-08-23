# tests/agent/test_insights_cache_read_pct.py
#
# B3 finding (2026-08-23) added `cache_read_pct` to each model entry in the
# Insights report so prompt-cache amplification surfaces directly in the
# Models Used table. These tests pin the field's presence and shape.

import time
from pathlib import Path

import pytest

from agent.insights import InsightsEngine
from hermes_state import SessionDB


@pytest.fixture()
def db(tmp_path):
    session_db = SessionDB(Path(tmp_path / "state.db"))
    yield session_db
    session_db.close()


def test_cache_read_pct_high_for_opus_shape(db):
    db.create_session(session_id="s1", source="desktop",
                      model="anthropic/claude-opus-5")
    db.update_token_counts(
        "s1",
        input_tokens=4000, output_tokens=1363176,
        cache_read_tokens=577109621, cache_write_tokens=33963425,
        model="anthropic/claude-opus-5",
        billing_provider="anthropic", api_call_count=2054,
    )
    db._conn.commit()

    report = InsightsEngine(db).generate(days=30)
    models = {m["model"]: m for m in report["models"]}
    assert "claude-opus-5" in models
    m = models["claude-opus-5"]
    assert "cache_read_pct" in m
    # 577M cache_read out of ~612M total = ~94.3%
    assert m["cache_read_pct"] > 90.0
    assert m["cache_read_pct"] < 100.0


def test_cache_read_pct_zero_for_no_cache_model(db):
    db.create_session(session_id="s2", source="cli", model="openai/gpt-4o")
    db.update_token_counts(
        "s2",
        input_tokens=5000, output_tokens=1000,
        cache_read_tokens=0, cache_write_tokens=0,
        model="openai/gpt-4o", billing_provider="openai",
        api_call_count=10,
    )
    db._conn.commit()

    report = InsightsEngine(db).generate(days=30)
    models = {m["model"]: m for m in report["models"]}
    assert "gpt-4o" in models
    assert models["gpt-4o"]["cache_read_pct"] == 0.0


def test_cache_read_pct_terminal_format_marks_high_amplification(db):
    db.create_session(session_id="s3", source="desktop",
                      model="anthropic/claude-opus-5")
    db.update_token_counts(
        "s3",
        input_tokens=4000, output_tokens=1363176,
        cache_read_tokens=577109621, cache_write_tokens=33963425,
        model="anthropic/claude-opus-5",
        billing_provider="anthropic", api_call_count=2054,
    )
    db._conn.commit()

    report = InsightsEngine(db).generate(days=30)
    text = InsightsEngine(db).format_terminal(report)
    # B3 finding: when cache_read_pct >= 50, append "(cache NN%)" to the
    # Models Used row so the signal is visible inline.
    assert "(cache 94%)" in text
