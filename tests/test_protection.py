"""Unit tests for the code-driven profit guards (src/protection.py).

These cover the pure decision logic only — no network / order placement.
"""

from src.config import ProfitProtectionConfig
from src.protection import decide_protection, should_block_chase


def _cfg(**overrides) -> ProfitProtectionConfig:
    base = dict(
        enabled=True,
        dry_run=False,
        breakeven_trigger_r=1.0,
        breakeven_fee_pct=0.0015,
        giveback_pct=0.35,
        min_favorable_excursion_pct=0.005,
        no_chase_enabled=True,
        post_win_cooldown_minutes=45.0,
        no_chase_buffer_pct=0.001,
    )
    base.update(overrides)
    return ProfitProtectionConfig(**base)


# ── decide_protection: breakeven ratchet ────────────────────────────────────────


def test_no_action_before_one_r():
    # Long up only +0.5R, stop still below: nothing to do yet.
    d = decide_protection(side_long=True, avg_entry=100.0, mark=102.5, sl_price=95.0, peak_fe=2.5, cfg=_cfg())
    assert d["action"] == "none"


def test_move_to_breakeven_at_one_r():
    # peak favourable excursion == initial risk (5 px) → ratchet stop to fee-adjusted breakeven.
    d = decide_protection(side_long=True, avg_entry=100.0, mark=104.0, sl_price=95.0, peak_fe=5.0, cfg=_cfg())
    assert d["action"] == "move_breakeven"
    assert d["stopPrice"] == 100.0 * 1.0015  # above entry by the fee buffer


def test_no_breakeven_when_stop_already_protective():
    # Stop already above entry (in profit) → no further ratchet needed.
    d = decide_protection(side_long=True, avg_entry=100.0, mark=106.0, sl_price=101.0, peak_fe=6.0, cfg=_cfg())
    assert d["action"] == "none"


def test_short_breakeven_symmetry():
    d = decide_protection(side_long=False, avg_entry=100.0, mark=96.0, sl_price=105.0, peak_fe=5.0, cfg=_cfg())
    assert d["action"] == "move_breakeven"
    assert d["stopPrice"] == 100.0 * (1 - 0.0015)  # below entry for a short


# ── decide_protection: give-back cap ─────────────────────────────────────────────


def test_giveback_close_after_real_run():
    # Ran to +10 px, gave back to +4 px (>50% of peak) → close to lock the gain.
    d = decide_protection(side_long=True, avg_entry=100.0, mark=104.0, sl_price=95.0, peak_fe=10.0, cfg=_cfg())
    assert d["action"] == "close"


def test_no_giveback_when_peak_too_small():
    # Peak of 0.2 px is below the 0.3% min FE on a 100-priced asset → ignore noise.
    d = decide_protection(side_long=True, avg_entry=100.0, mark=99.9, sl_price=95.0, peak_fe=0.2, cfg=_cfg())
    assert d["action"] == "none"


def test_giveback_disabled_when_pct_zero():
    d = decide_protection(side_long=True, avg_entry=100.0, mark=104.0, sl_price=95.0, peak_fe=10.0, cfg=_cfg(giveback_pct=0.0))
    # With give-back off it should fall through to breakeven (peak 10 >= 1R of 5).
    assert d["action"] == "move_breakeven"


# ── decide_protection: give-back arming at 1R (giveback_arm_r) ───────────────────


def test_giveback_not_armed_below_one_r_run():
    """Sub-1R wobble is the original SL's job — the cap must not book fee-scale scratch wins.
    Risk = 5 px; peak ran only 2 px (0.4R, but > the 0.5% pct floor) then retraced fully."""
    d = decide_protection(side_long=True, avg_entry=100.0, mark=100.1, sl_price=95.0, peak_fe=2.0, cfg=_cfg())
    assert d["action"] == "none"


def test_giveback_armed_after_one_r_run():
    # Risk = 5 px; ran to 6 px (1.2R) then gave back >35% → close.
    d = decide_protection(side_long=True, avg_entry=100.0, mark=102.0, sl_price=95.0, peak_fe=6.0, cfg=_cfg())
    assert d["action"] == "close"


def test_giveback_pct_arming_when_no_stop_known():
    # Without a live stop the 1R arming can't be computed → falls back to pct arming (old behavior).
    d = decide_protection(side_long=True, avg_entry=100.0, mark=100.5, sl_price=None, peak_fe=2.0, cfg=_cfg())
    assert d["action"] == "close"


def test_giveback_arm_r_zero_reverts_to_pct_arming():
    # arm_r=0 disables risk-based arming even with a stop present (legacy pct behavior).
    d = decide_protection(side_long=True, avg_entry=100.0, mark=100.1, sl_price=95.0, peak_fe=2.0, cfg=_cfg(giveback_arm_r=0.0))
    assert d["action"] == "close"


# ── Regression: the actual ETH incident (2026-06-07/08) ──────────────────────────


def test_eth_runner_gives_back_is_closed():
    """The DCA'd ETH long (avg ~1622) ran to ~1711 (+89 px) then reversed.
    With the guard, a retrace past 50% of that run locks profit instead of round-tripping.
    """
    d = decide_protection(side_long=True, avg_entry=1622.0, mark=1665.0, sl_price=1655.0, peak_fe=89.0, cfg=_cfg())
    assert d["action"] == "close"  # would have locked ~+43 px instead of stopping out red


def test_top_entry_with_no_run_takes_no_action():
    """The fresh long opened at the 1711.76 top never ran up; the ratchet correctly
    does nothing (this trade is the no-chase guard's job, not profit-lock's)."""
    d = decide_protection(side_long=True, avg_entry=1711.76, mark=1705.0, sl_price=1658.5, peak_fe=0.0, cfg=_cfg())
    assert d["action"] == "none"


# ── should_block_chase (P2) ─────────────────────────────────────────────────────


def test_block_relong_at_higher_price_after_win():
    # Closed a long at 1709; trying to re-buy at 1711.76 (the exact failure) → blocked.
    assert should_block_chase(close_type="CLOSE_LONG", exit_price=1709.0, new_side="buy", new_price=1711.76, buffer_pct=0.001) is True


def test_allow_relong_at_better_price():
    # Re-buying meaningfully below the exit (a real pullback) is allowed.
    assert should_block_chase(close_type="CLOSE_LONG", exit_price=1709.0, new_side="buy", new_price=1690.0, buffer_pct=0.001) is False


def test_block_reshort_at_lower_price_after_win():
    assert should_block_chase(close_type="CLOSE_SHORT", exit_price=100.0, new_side="sell", new_price=99.95, buffer_pct=0.001) is True


def test_no_block_on_opposite_direction():
    # Closed a long, now going short → not chasing; allowed.
    assert should_block_chase(close_type="CLOSE_LONG", exit_price=1709.0, new_side="sell", new_price=1711.0, buffer_pct=0.001) is False


def test_no_block_on_missing_prices():
    assert should_block_chase(close_type="CLOSE_LONG", exit_price=0.0, new_side="buy", new_price=1711.0, buffer_pct=0.001) is False
