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


def _probe(taken, bracket, family="fade_extreme"):
  return {"realizedR": taken, "setupFamily": family,
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
