"""Accounting for successful provider responses that omit usage metadata."""

from types import SimpleNamespace

from agent import turn_usage


class _RecordingDB:
    def __init__(self):
        self.calls = []

    def queue_token_counts(self, session_id, **kwargs):
        self.calls.append((session_id, kwargs))


def test_record_api_call_without_usage_preserves_truthful_zero_token_accounting():
    db = _RecordingDB()
    agent = SimpleNamespace(
        _session_db=db,
        _session_db_created=True,
        session_id="session-no-usage",
        session_api_calls=0,
        model="route-model",
        provider="route-provider",
        base_url="http://127.0.0.1:4356/v1",
    )

    helper = getattr(turn_usage, "_record_api_call_without_usage", None)
    assert helper is not None
    assert helper(agent) is True

    assert agent.session_api_calls == 1
    assert db.calls == [
        (
            "session-no-usage",
            {
                "model": "route-model",
                "billing_provider": "route-provider",
                "billing_base_url": "http://127.0.0.1:4356/v1",
                "cost_status": "unknown",
                "cost_source": "none",
                "api_call_count": 1,
            },
        )
    ]


def test_record_api_call_without_usage_updates_memory_without_session_store():
    agent = SimpleNamespace(
        _session_db=None,
        session_id="session-no-store",
        session_api_calls=0,
    )

    assert turn_usage._record_api_call_without_usage(agent) is False
    assert agent.session_api_calls == 1
