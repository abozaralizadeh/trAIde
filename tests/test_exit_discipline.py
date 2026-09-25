"""Exit discipline: scoring discretionary closes against the brackets they overrode.

Origin (2026-09-02): 16 positions were closed by the agent against 2 by the profit-lock, at a
13-minute median hold on brackets whose targets need hours. Replaying those 16 on real 1m klines,
letting the bracket run was worth +3.05R against the +0.42R actually taken — a 2.63R gap, larger than
the entire net loss over the same window. The bot measured entry quality and never measured this.
"""
import time

import pytest

from src.edge import exit_discipline_stats
from src.memory import MemoryStore
from src.regime import held_position_noise_pct


def _store(tmp_path) -> MemoryStore:
  return MemoryStore(str(tmp_path / "mem.json"))


def _probe(taken, bracket, family="fade_extreme", closed_by="agent"):
  """A resolved exit probe. An AGENT row carries a replayed stack equal to its bracket, because since
  2026-09-25 only stack-scored agent closes count toward the verdict (bracket-only rows are audit)."""
  row = {"realizedR": taken, "setupFamily": family, "closedBy": closed_by,
         "outcome": {"resolved": "take_profit", "bracketR": bracket}}
  if closed_by == "agent":
    row["stack"] = {"stackR": bracket, "resolvedBy": "take_profit", "source": "test"}
  return row


# --- the scorecard -------------------------------------------------------------------------------

def test_scorecard_withholds_a_verdict_until_it_has_evidence():
  assert exit_discipline_stats([])["verdict"] == "insufficient data"
  assert exit_discipline_stats([_probe(0.1, 1.8)] * 7)["verdict"] == "insufficient data"
  assert exit_discipline_stats([_probe(0.1, 1.8)] * 8)["verdict"] == "closes destroy value"


def test_scorecard_is_symmetric_and_endorses_closes_that_beat_their_bracket():
  """This must not be a one-way ratchet against closing: ducking a stop is a real skill and the
  measurement has to be able to say so, or it is a veto wearing a scoreboard's clothes."""
  good = exit_discipline_stats([_probe(-0.2, -1.0)] * 10)   # closed early, dodged the full stop
  assert good["verdict"] == "closes add value"
  assert good["deltaR"] == pytest.approx(8.0)
  assert good["beatBracket"] == 10
  bad = exit_discipline_stats([_probe(0.1, 1.8)] * 10)
  assert bad["verdict"] == "closes destroy value"
  assert bad["deltaR"] == pytest.approx(-17.0)
  assert bad["beatBracket"] == 0
  mixed = exit_discipline_stats([_probe(-0.2, -1.0)] * 5 + [_probe(0.1, 0.9)] * 5)
  assert mixed["verdict"] == "neutral"


def test_scorecard_ignores_probes_that_have_not_resolved():
  rows = [_probe(0.1, 1.8)] * 8 + [{"realizedR": 0.5, "outcome": {}}] * 20
  assert exit_discipline_stats(rows)["n"] == 8


# --- recording and settling ----------------------------------------------------------------------

def test_only_exits_short_of_the_bracket_are_worth_scoring(tmp_path):
  """A stop or target actually being hit is the bracket working, not a discretionary call."""
  m = _store(tmp_path)
  m.record_exit_probe("AAVE-USDT", "short", 132.35, 135.25, 127.10, 131.94, realized_r=0.14)
  assert len(m.exit_probes()) == 1
  for bad in [("", "short", 1, 2, 0.5, 1.5), ("X-USDT", "sideways", 1, 2, 0.5, 1.5),
              ("X-USDT", "short", 0, 2, 0.5, 1.5), ("X-USDT", "short", "nan", 2, 0.5, 1.5)]:
    m.record_exit_probe(*bad)
  assert len(m.exit_probes()) == 1  # only the well-formed one survived


def test_settle_resolves_against_the_bracket_both_ways(tmp_path):
  m = _store(tmp_path)
  # The real AAVE trade: short 132.35, stop 135.25, target 127.10, closed by hand at 131.94.
  m.record_exit_probe("AAVE-USDT", "short", 132.35, 135.25, 127.10, 131.94, realized_r=0.14)
  assert m.settle_exit_probes({"AAVE-USDT": 130.0}) == 0        # still between the levels
  assert m.settle_exit_probes({"AAVE-USDT": 126.5}) == 1        # target reached
  out = m.exit_probes()[0]["outcome"]
  assert out["resolved"] == "take_profit"
  assert out["bracketR"] == pytest.approx((132.35 - 127.10) / (135.25 - 132.35))
  assert m.settle_exit_probes({"AAVE-USDT": 126.0}) == 0        # already resolved, never re-scored

  m2 = _store(tmp_path / "b")
  m2.record_exit_probe("X-USDT", "long", 100.0, 90.0, 130.0, 101.0, realized_r=0.1)
  assert m2.settle_exit_probes({"X-USDT": 89.0}) == 1
  assert m2.exit_probes()[0]["outcome"] == {"resolved": "stop", "bracketR": -1.0,
                                            "resolvedTs": m2.exit_probes()[0]["outcome"]["resolvedTs"],
                                            # the market that resolved it, so the one-off re-resolve of
                                            # spot-settled rows never touches it (re-run safe)
                                            "priceSource": "futures_mark"}


def test_unresolved_probes_are_marked_to_market_after_expiry(tmp_path):
  """A trade that merely drifted must still contribute, or the sample keeps only the dramatic ones."""
  m = _store(tmp_path)
  m.record_exit_probe("X-USDT", "long", 100.0, 90.0, 130.0, 101.0, realized_r=0.1)
  data = m._read()
  data["exit_probes"][0]["ts"] = int(time.time()) - 9 * 3600
  m._write(data)
  assert m.settle_exit_probes({"X-USDT": 105.0}, expire_hours=8.0) == 1
  out = m.exit_probes()[0]["outcome"]
  assert out["resolved"] == "expired"
  assert out["bracketR"] == pytest.approx(0.5)   # +5 on a 10-wide stop


def test_exit_probes_survive_the_time_based_retention_sweep(tmp_path):
  """Learning data is count-capped, never clock-pruned — the same rule signal probes needed, for the
  same reason: evidence tied to the clock deadlocks during a quiet spell."""
  m = _store(tmp_path)
  m.record_exit_probe("X-USDT", "long", 100.0, 90.0, 130.0, 101.0, realized_r=0.1)
  data = m._read()
  data["exit_probes"][0]["ts"] = int(time.time()) - 400 * 86400
  m._write(data)
  m._write(m._prune(m._read()))   # _prune IS the retention sweep
  assert len(m.exit_probes()) == 1, "a 400-day-old exit probe must survive the clock cutoff"


# --- the trigger floor ---------------------------------------------------------------------------

def test_held_position_noise_floor_uses_the_trades_own_geometry():
  """AAVE was woken for re-decision on 0.50%/0.74%/0.80% moves while its stop stood 2.2% away."""
  band = held_position_noise_pct({"fillPrice": 132.35, "stopLossPrice": 135.25, "stopAtrMult": 2.5})
  assert band == pytest.approx(2.191 / 2.5, abs=0.01)
  assert band > 0.5   # strictly above the configured PRICE_CHANGE_TRIGGER_PCT floor
  for bad in (None, {}, {"fillPrice": 100, "stopLossPrice": 110},        # no stopAtrMult
              {"fillPrice": 0, "stopLossPrice": 1, "stopAtrMult": 2.5},
              {"fillPrice": 100, "stopLossPrice": 100, "stopAtrMult": 2.5}):
    assert held_position_noise_pct(bad) is None


# --- attribution: the model is judged only on the closes it actually made ---------------------------

def test_trailing_stop_exits_are_not_blamed_on_the_model():
  """The live bug (2026-09-19): of 30 scored 'discretionary closes', 16 were the code's trailing stop
  (-8.99R) and ONE was the model (+1.54R). The scoreboard said "closes destroy value, -12.41R" and the
  prompt told the model to let that number decide how readily it closes. Only agent closes may count."""
  rows = [_probe(0.2, 1.8, "continuation", closed_by="protection")] * 16 + [_probe(1.8, 0.26, closed_by="agent")]
  out = exit_discipline_stats(rows)
  assert out["n"] == 1, "only the model's own close is scored"
  assert out["deltaR"] == pytest.approx(1.54, abs=0.01)
  assert out["otherExits"]["protection"]["n"] == 16
  assert out["otherExits"]["protection"]["deltaR"] == pytest.approx(16 * (0.2 - 1.8), abs=0.01)


def test_legacy_rows_without_attribution_are_not_counted_as_the_models():
  """Rows recorded before attribution existed are mostly trailing-stop exits; they must not inflate
  or deflate the model's own record."""
  rows = [{"realizedR": 0.1, "setupFamily": "x", "outcome": {"resolved": "stop", "bracketR": 1.5}}] * 13
  out = exit_discipline_stats(rows)
  assert out["n"] == 0 and out["verdict"] == "insufficient data"
  assert out["otherExits"]["unattributed"]["n"] == 13


def test_agent_close_marker_round_trips(tmp_path):
  m = MemoryStore(str(tmp_path / "m.json"))
  assert m.recent_agent_close("SOL-USDT") is False
  m.note_agent_close("SOL-USDT")
  assert m.recent_agent_close("SOL-USDT") is True
  assert m.recent_agent_close("SOL-USDT", within_sec=0) in (True, False)   # boundary is not an error
  assert m.recent_agent_close("ETH-USDT") is False                         # per symbol
  m.record_exit_probe("SOL-USDT", "long", 100, 90, 130, 105, realized_r=0.5, closed_by="agent")
  assert m.exit_probes()[-1]["closedBy"] == "agent"


def test_the_marker_is_set_only_where_the_model_closes():
  """Structural guard: ProtectionManager places its closes directly and must never set the marker,
  and the tools layer must set it on BOTH futures market-order placement sites."""
  from pathlib import Path
  root = Path(__file__).resolve().parents[1] / "src"
  assert "note_agent_close" not in (root / "protection.py").read_text()
  assert (root / "tools.py").read_text().count("memory.note_agent_close(spot_symbol)") == 2


# --- a flip must not stamp the closed side with the new side's lifecycle -------------------------

def test_flip_close_does_not_borrow_the_new_positions_lifecycle():
  """H-USDT, 2026-09-22 06:04: the model closed a LONG and opened a SHORT in the same run. When it
  logged the long's close, the live book already held the short, so the close row was stamped with the
  short's openTime and side — unmatchable to its own fill: no prices, an empty dashboard card, and no
  exit probe. The row is then invisible to the very scoreboard meant to judge the model's closes."""
  from src.tools import lifecycle_for_close
  short_now = {"currentQty": -5, "id": "p2", "openTime": 1790055100087}
  assert lifecycle_for_close("close_long", short_now) == {}
  assert lifecycle_for_close("futures_close_long", short_now) == {}
  long_now = {"currentQty": 5, "id": "p1", "openTime": 1790050000000}
  assert lifecycle_for_close("close_long", long_now) == {
    "position_id": "p1", "position_open_time": 1790050000000, "position_side": "long"}
  assert lifecycle_for_close("close_short", long_now) == {}
  assert lifecycle_for_close("close_short", short_now)["position_side"] == "short"


def test_side_less_close_actions_keep_the_old_behaviour():
  """'futures_close' / 'close_position' name no side, so the live book is the only evidence there is."""
  from src.tools import lifecycle_for_close
  pos = {"currentQty": -3, "positionId": "p9", "openingTimestamp": 42}
  assert lifecycle_for_close("futures_close", pos) == {
    "position_id": "p9", "position_open_time": 42, "position_side": "short"}
  assert lifecycle_for_close("close_long", None) == {}
  assert lifecycle_for_close("close_long", {"currentQty": 0}) == {}   # flat book: nothing to vouch


# --- regime tag on exit probes, and the trail split by regime -----------------------------------

def test_exit_probe_stores_the_entrys_regime_tag(tmp_path):
  m = MemoryStore(str(tmp_path / "m.json"))
  m.record_exit_probe("X-USDT", "long", 100, 90, 130, 105, realized_r=0.5, closed_by="protection",
                      regime={"market_regime": "trending", "strength": "strong", "daily_atr_pct": 6.8})
  m.record_exit_probe("Y-USDT", "long", 100, 90, 130, 105, realized_r=0.5, closed_by="protection",
                      regime="junk")
  rows = m.exit_probes()
  assert rows[0]["regime"] == {"market_regime": "trending", "strength": "strong"}   # only the two keys
  assert rows[1]["regime"] is None


def test_trail_record_is_split_by_regime_and_only_for_the_trail():
  """The trail is right in chop and wrong in a trend. The split is what lets a regime-adaptive trail
  be justified — or refused — on evidence from BOTH regimes rather than one."""
  def p(taken, bracket, who, regime):
    return {"realizedR": taken, "setupFamily": "continuation", "closedBy": who,
            "regime": regime, "outcome": {"resolved": "take_profit", "bracketR": bracket}}
  rows = ([p(0.3, 1.8, "protection", {"market_regime": "trending", "strength": "strong"})] * 4
          + [p(0.2, -1.0, "protection", {"market_regime": "ranging", "strength": "weak"})] * 3
          + [p(0.9, 0.1, "agent", {"market_regime": "trending", "strength": "strong"})]       # model close: excluded
          + [p(0.1, 1.0, "protection", None)])                                                  # untagged: excluded
  out = exit_discipline_stats(rows)
  tbr = out["trailByRegime"]
  assert set(tbr) == {"trending/strong", "ranging/weak"}
  assert tbr["trending/strong"]["n"] == 4 and tbr["trending/strong"]["deltaR"] == pytest.approx(4 * (0.3 - 1.8))
  assert tbr["ranging/weak"]["n"] == 3 and tbr["ranging/weak"]["deltaR"] == pytest.approx(3 * (0.2 + 1.0))
  assert out["otherExits"]["protection"]["n"] == 8     # the untagged one still counts as a trail exit


# --- the benchmark is the live exit STACK, not the bare bracket (2026-09-25) ----------------------
# The system never leaves a position on its bare bracket: breakeven, the noise-band trail and the carry
# hold manage it every poll. On the 5 attributed agent closes at the time the bracket read -6.23R and the
# replayed stack about -3.5R (DASH/INJ: trail exits near +0.3..0.5R before a TP reached hours later).

def _stack_probe(taken, bracket, stack=None, *, family="funding_carry", closed_by="agent", inputs=True,
                 counter=None, htf=None, resolved="take_profit"):
  row = {"realizedR": taken, "setupFamily": family, "closedBy": closed_by,
         "outcome": {"resolved": resolved, "bracketR": bracket}, "counterAtEntry": counter,
         "htfAligned": htf}
  if inputs:
    row.update({"fillTs": 1_790_000_000, "initRiskPx": 0.5})
  if stack is not None:
    row["stack"] = {"stackR": stack, "resolvedBy": "trail_close", "resolvedTs": 1_790_030_000,
                    "source": "live_1m_replay"}
  return row


def test_agent_closes_are_scored_against_stackR_and_bracket_only_rows_are_audit():
  rows = [_stack_probe(0.09, 1.70, 0.50), _stack_probe(-0.25, 1.87, 0.33),   # DASH / INJ shape
          _stack_probe(-0.2, -1.0, inputs=False)]                            # recorded before the replay
  out = exit_discipline_stats(rows)
  assert out["n"] == 2                                                       # ONE comparator in the verdict
  assert out["stackScored"] == 2 and out["legacyBracketScored"] == 1
  assert out["benchmarkR"] == pytest.approx(0.50 + 0.33)
  assert out["stackR"] == pytest.approx(0.83)
  assert out["bracketR"] == pytest.approx(1.70 + 1.87)                       # kept for audit only
  assert out["deltaR"] == pytest.approx((0.09 - 0.25) - (0.50 + 0.33))
  assert out["legacyDeltaR"] == pytest.approx(0.8)                           # audit, not in deltaR
  assert out["stackVsBracket"] == {"n": 2, "deltaR": pytest.approx((0.50 + 0.33) - (1.70 + 1.87))}
  assert out["beatBenchmark"] == 0 and out["beatBracket"] == 0


def test_the_verdict_follows_the_stack_not_the_bracket():
  """Eight closes that each gave up a distant TP the trail would never have reached: on the bracket
  that reads 'closes destroy value', against the system that would actually have run it is level —
  and bracket-only rows give NO verdict at all rather than the bracket's."""
  rows = [_stack_probe(0.30, 1.80, 0.31)] * 8
  assert exit_discipline_stats(rows)["verdict"] == "neutral"
  legacy = exit_discipline_stats([_stack_probe(0.30, 1.80, inputs=False)] * 8)
  assert legacy["verdict"] == "insufficient data" and legacy["n"] == 0 and legacy["legacyBracketScored"] == 8


def test_legacy_bracket_rows_never_unlock_or_steer_the_first_verdict():
  """C3 review: 5 legacy rows + 3 stack rows read n=8 and 'closes destroy value' (-0.575R/trade) while
  the 3 stack rows alone were +0.017R/trade — two benchmarks blended into one verdict."""
  legacy = [_stack_probe(t, b, inputs=False) for t, b in
            ((0.09, 1.70), (-0.25, 1.87), (0.38, 1.88), (0.54, -1.0), (-0.8, 1.2))]
  stacked = [_stack_probe(0.2, 0.5, 0.18), _stack_probe(-0.1, -1.0, -0.15), _stack_probe(0.3, 0.9, 0.3)]
  out = exit_discipline_stats(legacy + stacked)
  assert out["n"] == 3 and out["verdict"] == "insufficient data"
  assert out["deltaRPerTrade"] == pytest.approx(((0.2 - 0.1 + 0.3) - (0.18 - 0.15 + 0.3)) / 3, abs=1e-4)
  assert out["legacyBracketScored"] == 5 and "not counted" in out["note"]


def test_a_row_awaiting_its_replay_is_left_out_not_scored_on_the_bracket():
  """Blending benchmarks mid-flight would flip the verdict as rows mature; the row waits instead."""
  out = exit_discipline_stats([_stack_probe(0.1, 1.8), _stack_probe(0.1, 1.8, 0.2)])
  assert out["n"] == 1 and out["stackPending"] == 1
  gave_up = _stack_probe(0.1, 1.8)
  gave_up["stack"] = {"stackR": None, "resolvedBy": "unavailable"}
  out = exit_discipline_stats([gave_up])
  assert out["n"] == 0 and out["stackUnavailable"] == 1 and out["stackPending"] == 0
  assert out["legacyBracketScored"] == 0
  # Five given-up replays next to three stack rows: still n == 3, never a bracket-driven verdict.
  out = exit_discipline_stats([dict(gave_up) for _ in range(5)] + [_stack_probe(0.1, 1.8, 0.2)] * 3)
  assert out["n"] == 3 and out["stackUnavailable"] == 5 and out["verdict"] == "insufficient data"


def test_a_stack_scored_row_counts_even_when_the_bracket_went_unmeasured():
  row = _stack_probe(0.2, None, 0.4, resolved="unmeasured")
  out = exit_discipline_stats([row])
  assert out["n"] == 1 and out["deltaR"] == pytest.approx(-0.2) and out["stackVsBracket"]["n"] == 0


def test_other_exits_stay_on_the_bracket_split_by_how_the_bracket_resolved():
  """For the trail's own exits the question IS trail-vs-bracket; against the stack a trail exit would
  score ~0 by construction and the evidence a regime-adaptive trail waits for would vanish."""
  rows = ([_stack_probe(0.3, 1.8, 0.3, closed_by="protection")] * 3
          + [_stack_probe(0.1, -1.0, closed_by="protection", resolved="stop")] * 2
          + [_stack_probe(0.2, 0.5, closed_by="protection", resolved="expired")])
  other = exit_discipline_stats(rows)["otherExits"]["protection"]
  assert other["n"] == 6
  assert other["deltaR"] == pytest.approx(3 * (0.3 - 1.8) + 2 * (0.1 + 1.0) + (0.2 - 0.5))
  assert other["byResolution"]["take_profit"] == {"n": 3, "deltaR": pytest.approx(-4.5)}
  assert other["byResolution"]["stop"] == {"n": 2, "deltaR": pytest.approx(2.2)}
  assert other["byResolution"]["expired"]["n"] == 1


def test_closes_are_split_by_the_entrys_own_bias_with_n_before_any_verdict():
  """Premise-opposition closes (15m AND 1h already against the side at entry) lost on DASH/KCS and
  helped on G/XMR — n=5 in one rally. The split lets the model's own record decide, shown with n."""
  rows = [_stack_probe(0.09, 1.7, 0.5, counter=True, htf=True),
          _stack_probe(0.38, 1.88, 1.88, counter=True, htf=True),
          _stack_probe(0.54, -1.0, -1.0, family="funding_carry", counter=True, htf=False),
          _stack_probe(-0.25, 1.87, 0.33, family="continuation", counter=False, htf=True),
          _stack_probe(-0.8, -1.0, -1.0, family="continuation")]
  out = exit_discipline_stats(rows)
  assert out["verdict"] == "insufficient data"
  assert out["byFamily"]["funding_carry"]["n"] == 3 and out["byFamily"]["continuation"]["n"] == 2
  assert out["byCounterAtEntry"]["true"]["n"] == 3
  assert out["byCounterAtEntry"]["true"]["deltaR"] == pytest.approx((0.09 - 0.5) + (0.38 - 1.88) + (0.54 + 1.0))
  assert out["byCounterAtEntry"]["false"]["n"] == 1
  assert out["byCounterAtEntry"]["untagged"]["n"] == 1
  assert out["byHtfAligned"]["true"]["n"] == 3 and out["byHtfAligned"]["false"]["n"] == 1


def test_exit_probe_records_the_stack_inputs_and_entry_bias_tags(tmp_path):
  m = MemoryStore(str(tmp_path / "m.json"))
  m.record_exit_probe("KCS-USDT", "long", 13.8, 13.62, 14.14, 13.87, realized_r=0.38, closed_by="agent",
                      setup_family="funding_carry", fill_ts=1_790_244_000, init_risk_px=0.18,
                      noise_band_r=0.378, hold_until_ts=1_790_265_600,
                      entry_bias={"15m": "bearish", "1h": "bearish", "4h": "bullish", "1D": "bullish"},
                      counter_at_entry=True, htf_aligned=True)
  m.record_exit_probe("X-USDT", "long", 100, 90, 130, 105, realized_r=0.5, closed_by="agent",
                      fill_ts="junk", init_risk_px=float("nan"), counter_at_entry="yes")
  kcs, x = m.exit_probes()
  assert kcs["fillTs"] == 1_790_244_000 and kcs["initRiskPx"] == pytest.approx(0.18)
  assert kcs["noiseBandR"] == pytest.approx(0.378) and kcs["holdUntilTs"] == 1_790_265_600
  assert kcs["entryBias"]["15m"] == "bearish" and kcs["counterAtEntry"] is True and kcs["htfAligned"] is True
  assert x["fillTs"] is None and x["initRiskPx"] is None and x["counterAtEntry"] is None


def test_the_stack_result_is_stored_once_on_the_matching_probe(tmp_path):
  m = MemoryStore(str(tmp_path / "m.json"))
  m.record_exit_probe("DASH-USDT", "long", 55.0, 54.4, 56.0, 55.05, realized_r=0.09, closed_by="agent",
                      fill_ts=1_790_202_000, init_risk_px=0.6)
  ts = m.exit_probes()[0]["ts"]
  assert m.set_exit_probe_stack("DASHUSDTM", ts + 1, 0.5, "trail_close", 1, "x") is False   # wrong ts
  assert m.set_exit_probe_stack("DASHUSDTM", ts, 0.5, "trail_close", 1_790_220_960, "live_1m_replay",
                                pre_close_exit_suppressed=False) is True
  assert m.set_exit_probe_stack("DASH-USDT", ts, 9.9, "take_profit", 2, "again") is False    # written once
  stack = m.exit_probes()[0]["stack"]
  assert stack["stackR"] == pytest.approx(0.5) and stack["resolvedBy"] == "trail_close"
  assert stack["source"] == "live_1m_replay" and stack["preCloseExitSuppressed"] is False
  assert exit_discipline_stats(m.exit_probes())["stackScored"] == 1


# --- the trail's record split by MARKET state (2026-09-25) ------------------------------------------
# The per-symbol regime tag read 'trending/strong' on every tagged trail exit, so trailByRegime's chop
# row could never fill. The entry's breadth24 (share of liquid perps up over 24h) can.

def _ms_probe(taken, bracket, breadth, who="protection"):
  row = {"realizedR": taken, "setupFamily": "continuation", "closedBy": who,
         "regime": {"market_regime": "trending", "strength": "strong"},
         "outcome": {"resolved": "take_profit", "bracketR": bracket}}
  if breadth is not None:
    row["marketState"] = {"breadth24": breadth}
  return row


def test_trail_record_is_split_by_rolling_breadth_terciles_with_n():
  rows = ([_ms_probe(0.2, -1.0, b) for b in (0.05, 0.10, 0.15)]        # low breadth: trail saved 1.2R each
          + [_ms_probe(0.3, 0.5, b) for b in (0.40, 0.50, 0.55)]
          + [_ms_probe(0.3, 1.8, b) for b in (0.90, 1.00)]             # high breadth: trail left 1.5R each
          + [_ms_probe(0.1, 1.0, None)]                                 # legacy row: untagged, still counted
          + [_ms_probe(0.9, 0.1, 0.95, who="agent")]                   # the model's close: not the trail
          + [_ms_probe(0.5, 1.8, 0.99, who=None)])                     # pre-attribution row: not the trail
  out = exit_discipline_stats(rows)
  tbm = out["trailByMarketState"]
  # Cuts are the retained probes' OWN terciles (nearest rank over all 10 tagged rows, the agent's and the
  # unattributed close included in the distribution), not fixed edges.
  assert tbm["cuts"] == [0.4, 0.9]
  b = tbm["buckets"]
  assert b["low"]["n"] == 4 and b["low"]["deltaR"] == pytest.approx(3 * 1.2 - 0.2)   # 0.05..0.15 + 0.40
  assert b["mid"]["n"] == 3 and b["mid"]["deltaR"] == pytest.approx(2 * -0.2 - 1.5)  # 0.50, 0.55, 0.90
  assert b["high"]["n"] == 1 and b["high"]["deltaR"] == pytest.approx(-1.5)         # 1.00
  assert b["untagged"]["n"] == 1
  assert sum(v["n"] for v in b.values()) == out["otherExits"]["protection"]["n"]
  # trailByRegime is unchanged (every row still reads trending/strong).
  assert list(out["trailByRegime"]) == ["trending/strong"]


def test_the_trail_record_is_also_split_by_btc_daily_adx_so_a_chop_row_can_fill():
  """C6 review: breadth is DIRECTION. An August-style high-breadth chop (BTC daily ADX 10-21) spreads
  across every breadth tercile, so the adaptive-trail decision's chop row could never fill. The entry's
  BTC daily ADX (stamped on every probe) separates it; nested so the prompt filter still drops it."""
  def p(taken, bracket, breadth, adx):
    row = _ms_probe(taken, bracket, breadth)
    row["marketState"]["btcDailyAdx"] = adx
    return row
  chop = [p(0.2, -1.0, b, a) for b, a in zip((0.5, 0.6, 0.7, 0.8, 0.85, 0.55), (10, 12, 14, 16, 18, 21))]
  rally = [p(0.3, 1.8, b, a) for b, a in zip((0.65, 0.7, 0.8, 0.9, 0.75, 0.85), (30, 32, 34, 36, 38, 40))]
  mid = [p(0.1, 0.1, b, a) for b, a in zip((0.6, 0.7, 0.8, 0.9, 0.6, 0.75), (22, 24, 25, 26, 27, 28))]
  out = exit_discipline_stats(chop + rally + mid)["trailByMarketState"]
  by_breadth = out["buckets"]
  assert all(k in by_breadth for k in ("low", "mid", "high"))
  adx = out["byBtcDailyAdx"]
  assert adx["cuts"] == [21, 28]
  assert adx["buckets"]["low"] == {"n": 6, "deltaR": pytest.approx(6 * 1.2)}     # the chop cohort, alone
  assert adx["buckets"]["high"] == {"n": 6, "deltaR": pytest.approx(6 * -1.5)}   # the rally cohort, alone
  assert adx["buckets"]["mid"]["n"] == 6
  from src.agent import _exit_discipline_for_prompt
  assert "trailByMarketState" not in _exit_discipline_for_prompt(exit_discipline_stats(chop + rally))


def test_too_few_tagged_rows_have_no_cuts():
  out = exit_discipline_stats([_ms_probe(0.2, -1.0, 0.3), _ms_probe(0.2, -1.0, None)])
  assert out["trailByMarketState"]["cuts"] is None
  assert out["trailByMarketState"]["buckets"] == {"untagged": {"n": 2, "deltaR": pytest.approx(2.4)}}


def test_main_stamps_the_entry_and_exit_market_state_on_the_exit_probe():
  """Wiring: the recorder gets the ENTRY's stamp (what the split keys on) and the loop's reading."""
  import inspect
  import src.main as main_mod
  src = inspect.getsource(main_mod.trading_loop)
  call = src[src.index("memory.record_exit_probe("):src.index("except Exception as exc:\n          logger.warning(\"EXIT PROBE")]
  assert 'market_state=_ctx.get("marketState")' in call
  assert "market_state_at_exit=_market_state.current()" in call
