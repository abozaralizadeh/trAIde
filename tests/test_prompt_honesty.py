"""The prompt must describe the gates exactly as code enforces them, the same for both sides.

2026-09-28 analysis (docs/analysis/2026-09-28-why-few-trades-no-shorts.md): the prompt called counter-daily
trades "BLOCKED at the code level ... NOT optional" while the order path admits a confirmed reversal and
declared playbooks past that gate. `daily_opposing` made 0 hard refusals over Sep 27-28 and the model still
self-censored every counter-daily short. The model obeys the prompt, not the gate.
"""
from pathlib import Path
from types import SimpleNamespace

from src.agent import _daily_gate_rules
from src.analytics import summarize_multi_timeframe
from src.config import RegimeConfig

SRC = Path(__file__).resolve().parents[1] / "src"


def _regime(**overrides) -> RegimeConfig:
  base = dict(
    throttle_enabled=True, caution_min_confidence=0.75, caution_size_factor=0.6,
    trend_shorts_enabled=True, trend_short_min_confidence=0.78, trend_short_require_15m=True,
  )
  base.update(overrides)
  return RegimeConfig(**base)


def _rules(**overrides) -> str:
  return _daily_gate_rules(SimpleNamespace(regime=_regime(**overrides)))


class TestDailyGateText:
  def test_the_prompt_no_longer_claims_counter_daily_trades_are_unconditionally_blocked(self):
    agent_src = (SRC / "agent.py").read_text()
    assert "This is NOT optional" not in agent_src
    assert "Counter-daily trades are BLOCKED at the code level" not in agent_src
    assert "Trade WITH the daily trend, not against it" not in agent_src
    assert "_daily_gate_rules(cfg)" in agent_src

  def test_thresholds_come_from_config_not_a_hard_coded_080(self):
    text = _rules(reversal_short_min_confidence=0.83, reversal_long_min_confidence=0.86, trend_short_min_confidence=0.71)
    assert "confidence >= 0.83" in text and "confidence >= 0.86" in text and "confidence >= 0.71" in text
    assert "0.80" not in text

  def test_it_says_the_label_is_completed_bars_only(self):
    assert "COMPLETED UTC daily candles only" in _rules()

  def test_every_enabled_route_is_named(self):
    text = _rules()
    assert "confirmed REVERSAL" in text
    assert "fade_extreme" in text and "<= 30 to buy" in text and ">= 70 to sell" in text
    assert "breakout/range_edge/funding_carry/macro_event" in text

  def test_a_disabled_route_is_never_advertised(self):
    text = _rules(
      reversal_shorts_enabled=False, reversal_longs_enabled=False, fade_extreme_enabled=False,
      declared_setups_enabled=False, trend_shorts_enabled=False,
    )
    assert "REVERSAL" not in text and "fade_extreme" not in text and "declared" not in text
    assert "continuation SHORT" not in text
    assert "none are enabled" in text

  def test_the_15m_requirement_follows_config(self):
    assert "when 1h has turned bearish" in _rules(reversal_short_require_15m=False)
    assert "when 1h AND 15m have turned bearish" in _rules(reversal_short_require_15m=True)

  def test_it_is_worded_the_same_for_both_sides(self):
    text = _rules()
    assert "a SHORT under a bullish daily" in text and "a LONG under a bearish daily" in text
    assert "Same rule for both sides" in text

  def test_exhausted_dailies_are_said_to_gate_neither_side_on_direction(self):
    # daily_bias reads 'neutral' when exhausted; the model cited a "bullish daily" against shorts on
    # exhausted coins (BCH, FET, SUI, RENDER on Sep 28) where no daily gate existed.
    assert "refuses NEITHER side" in _rules()

  def test_the_retry_text_does_not_forbid_the_hatches(self):
    agent_src = (SRC / "agent.py").read_text()
    assert "Do NOT retry the same direction — trade WITH the daily trend" not in agent_src
    assert "again only once a route's condition is genuinely true" in agent_src


class TestAnalyticsHint:
  def _summary(self, daily, intraday, rsi=55, adx=30):
    return summarize_multi_timeframe([
      {"interval": "1day", "trend_bias": daily, "rsi": rsi, "adx": adx, "volatility": "normal"},
      {"interval": "4hour", "trend_bias": intraday, "volatility": "normal"},
      {"interval": "1hour", "trend_bias": intraday, "volatility": "normal"},
      {"interval": "15min", "trend_bias": intraday, "volatility": "normal"},
    ])

  def test_daily_gate_hint_names_the_refused_side_and_the_routes(self):
    out = self._summary("bullish", "bearish")
    hint = out["entry_hint"]
    assert out["daily_gate_applied"] is True and out["timeframe_conflict"] is True
    assert "Do NOT open counter-daily trades" not in hint
    assert "A plain short is refused in code" in hint
    assert "confirmed reversal" in hint and "declared playbook" in hint
    assert "completed daily bars" in hint

  def test_mirror_for_a_bearish_daily(self):
    assert "A plain long is refused in code" in self._summary("bearish", "bullish")["entry_hint"]

  def test_exhausted_bullish_hint_does_not_claim_a_daily_gate_on_shorts(self):
    out = self._summary("bullish", "bearish", rsi=75)
    assert out["daily_exhausted"] is True and out["daily_bias"] == "neutral"
    assert "does not apply to shorts" in out["entry_hint"]


class TestFadeSetupHonesty:
  def test_tf_conflict_predicate_truth_table(self):
    from src.regime import tf_conflict_opposes
    assert tf_conflict_opposes(True, "bearish", "buy") is True
    assert tf_conflict_opposes(True, "bullish", "sell") is True
    assert tf_conflict_opposes(True, "bullish", "long") is False
    assert tf_conflict_opposes(True, "bearish", "short") is False
    assert tf_conflict_opposes(True, "neutral", "buy") is False
    assert tf_conflict_opposes(False, "bearish", "buy") is False

  def test_the_prompt_no_longer_calls_fade_setup_ready_made(self):
    agent_src = (SRC / "agent.py").read_text()
    assert "ready-made fade_extreme candidate" not in agent_src
    assert "tfConflictRefuses" in agent_src


class TestSymmetricWording:
  """Mirror pairs: every side-specific instruction has its opposite-side twin (2026-09-28: the prompt's
  'Trade STRENGTH', RSI 65-vs-40 confirmation, the 40-65 strong band and the ONDO case were written from
  the long side only). Hygiene, not a claim that shorts pay — the rally longs were earned under the old text."""

  def _src(self) -> str:
    return (SRC / "agent.py").read_text()

  def test_trend_bullet_names_both_directions(self):
    src = self._src()
    assert "- Trade STRENGTH." not in src
    assert "relative strength for a long, relative weakness for a short" in src

  def test_rejection_confirmation_uses_one_mirrored_rsi_pair(self):
    src = self._src()
    assert "RSI > 65" in src and "RSI < 35.\\n" in src
    assert "RSI < 40 or bullish divergence" not in src

  def test_strong_setup_band_is_given_per_side(self):
    assert "RSI 40–65 for a long / 35–60 for a" in self._src()

  def test_entry_headers_are_mirrored(self):
    src = self._src()
    assert "SHORT entry — entry_price at or above current price" in src
    assert "LONG entry — entry_price at or below current price" in src

  def test_the_ondo_case_has_its_mirror_and_keeps_the_intact_trend_gate(self):
    src = self._src()
    assert "BREAKDOWN THAT WON'T BOUNCE" in src
    assert "gated to an intact trend (daily+intraday aligned)" in src

  def test_research_scan_says_momentum_covers_decliners(self):
    src = self._src()
    assert "largest ABSOLUTE 24h" in src
    assert "in a bearish \"\n      \"BTC daily, 'losers'/'short'" not in src
