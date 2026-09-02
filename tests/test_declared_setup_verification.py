"""A declared playbook with an OBJECTIVE trigger must actually have that trigger present.

Live case, 2026-09-02 (XMR-USDT): funding was 0.0523%/8h against a threshold of 0.0700% derived from
measured round-trip cost, so `funding_carry_setup` returned None and "FUNDING CARRY AVAILABLE" had
never once been logged. A short was nonetheless declared `funding_carry`, admitted past the 1h gate
on the strength of the label — 1h/4h/15m were all bullish against it — and closed at -0.6R. The carry
it stood to earn was 0.0523% against a 1.89% stop, a 1:36 ratio: a directional short wearing a carry
label, while the `funding_carry` scoreboard was scored on it.

The declaration carve-out is right for `breakout`/`range_edge` — no computable trigger exists, so the
model's word is the only evidence there is. It must not become a universal gate-bypass token.
"""
import pytest

from src.regime import funding_carry_setup, verify_declared_setup


COST = 2.0 * (0.0006 + 0.0001)   # taker + measured slippage, the same basis the entry gates use


class TestFundingCarryMustActuallyPay:
  def test_the_live_xmr_rate_does_not_qualify(self):
    """0.0523%/8h against a 0.0700% threshold — the setup detector says no."""
    assert funding_carry_setup(0.000523, COST) is None
    reason = verify_declared_setup("funding_carry", side="sell", funding_setup=None)
    assert reason is not None
    assert "not extreme enough" in reason

  def test_a_qualifying_rate_on_the_paid_side_is_accepted(self):
    setup = funding_carry_setup(0.0012, COST)          # longs pay shorts
    assert setup is not None and setup["side"] == "sell"
    assert verify_declared_setup("funding_carry", side="sell", funding_setup=setup) is None

  def test_the_declared_side_must_be_the_side_that_receives(self):
    """Positive funding pays the SHORT. A long declaring carry would be paying it away."""
    setup = funding_carry_setup(0.0012, COST)
    reason = verify_declared_setup("funding_carry", side="buy", funding_setup=setup)
    assert reason is not None
    assert "PAID" in reason
    # ...and the mirror case: negative funding pays the long.
    neg = funding_carry_setup(-0.0012, COST)
    assert neg["side"] == "buy"
    assert verify_declared_setup("funding_carry", side="buy", funding_setup=neg) is None
    assert verify_declared_setup("funding_carry", side="sell", funding_setup=neg) is not None

  def test_threshold_tracks_measured_cost_rather_than_a_constant(self):
    """As execution gets cheaper the bar falls with it — nothing to re-tune by hand."""
    rate = 0.0004
    assert funding_carry_setup(rate, COST) is None            # 0.04% < 0.07% threshold
    cheaper = 2.0 * (0.0002 + 0.00002)                        # maker-ish fills
    assert funding_carry_setup(rate, cheaper) is not None      # same rate now clears it


class TestMacroEventMustFollowARelease:
  def test_only_the_after_window_qualifies(self):
    assert verify_declared_setup("macro_event", side="sell", macro_window={"phase": "after"}) is None
    # 'before' is the blackout, not a playbook; no window at all is not one either.
    assert verify_declared_setup("macro_event", side="sell", macro_window={"phase": "before"}) is not None
    assert verify_declared_setup("macro_event", side="sell", macro_window=None) is not None
    assert verify_declared_setup("macro_event", side="sell") is not None


class TestJudgementPlaybooksAreUnaffected:
  def test_families_with_no_computable_trigger_still_pass_on_the_declaration(self):
    """This is the whole point of the carve-out and must not regress: breakout/range_edge are
    judgement calls that could never reach the book if code demanded evidence for them."""
    for fam in ("breakout", "range_edge", "continuation", "fade_extreme", "other", None, ""):
      assert verify_declared_setup(fam, side="buy") is None
      assert verify_declared_setup(fam, side="sell", funding_setup=None, macro_window=None) is None

  def test_case_and_whitespace_do_not_smuggle_a_mechanical_family_through(self):
    for label in ("funding_carry", "FUNDING_CARRY", "  Funding_Carry  "):
      assert verify_declared_setup(label, side="sell", funding_setup=None) is not None
    for label in ("macro_event", "MACRO_EVENT", " Macro_Event "):
      assert verify_declared_setup(label, side="sell", macro_window=None) is not None
