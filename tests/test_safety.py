from src.safety import TradingSafetyState


def test_entry_authority_requires_current_matching_run():
    state = TradingSafetyState()
    assert state.authorize_run() is None
    state.refresh(71.0)
    token = state.authorize_run()
    assert token
    assert state.check(token, max_age_sec=60)["allowed"] is True
    assert state.check("wrong", max_age_sec=60)["allowed"] is False
    state.revoke_run(token, "timeout")
    assert state.check(token, max_age_sec=60)["allowed"] is False


def test_invalidation_and_shutdown_revoke_entries():
    state = TradingSafetyState()
    state.refresh(71.0)
    token = state.authorize_run()
    state.invalidate("snapshot incomplete")
    state.refresh(70.0)
    assert state.check(token, max_age_sec=60)["allowed"] is False
    fresh = state.authorize_run()
    assert state.check(fresh, max_age_sec=60)["allowed"] is True
    state.begin_shutdown()
    assert state.check(fresh, max_age_sec=60)["allowed"] is False
    assert state.authorize_run() is None
