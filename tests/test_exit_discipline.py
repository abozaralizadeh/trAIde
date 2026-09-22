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
  return {"realizedR": taken, "setupFamily": family, "closedBy": closed_by,
          "outcome": {"resolved": "take_profit", "bracketR": bracket}}


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
                                            "resolvedTs": m2.exit_probes()[0]["outcome"]["resolvedTs"]}


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
