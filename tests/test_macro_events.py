"""Scheduled macro releases: a blackout on NEW risk before, a scored playbook after.

The one news input that does not require latency — CPI/FOMC/NFP dates are published a year ahead, so
this is a calendar, not a feed. Windows default to an hour either side because that is where BTC's
sensitivity to CPI surprises was measured to concentrate through 2026, and single prints moved it
4-8% intraday. The effect is decaying, which is why only the RISK half is enforced in code; the
tradeable half is a normal explore-sized family that de-sizes itself if it does not pay.
"""
import time

import pytest

from src.config import load_config
from src.edge import SETUP_FAMILIES
from src.memory import MemoryStore
from src.regime import macro_event_entry_block, macro_event_window


def _ev(minutes_from_now, name="US CPI", impact="high", now=None):
  return {"name": name, "ts": (now or time.time()) + minutes_from_now * 60, "impact": impact}


# --- the window ----------------------------------------------------------------------------------

def test_window_phases_split_before_and_after_the_release():
  now = 1_700_000_000.0
  assert macro_event_window([_ev(30, now=now)], now)["phase"] == "before"
  assert macro_event_window([_ev(-30, now=now)], now)["phase"] == "after"
  assert macro_event_window([_ev(90, now=now)], now) is None      # outside the hour
  assert macro_event_window([_ev(-90, now=now)], now) is None


def test_only_high_impact_events_count_by_default():
  now = 1_700_000_000.0
  assert macro_event_window([_ev(30, impact="medium", now=now)], now) is None
  assert macro_event_window([_ev(30, impact="high", now=now)], now) is not None


def test_nearest_event_wins_when_two_are_close():
  now = 1_700_000_000.0
  w = macro_event_window([_ev(50, name="FOMC", now=now), _ev(10, name="US CPI", now=now)], now)
  assert w["event"]["name"] == "US CPI"


def test_a_missing_or_broken_calendar_degrades_to_ordinary_trading():
  """The failure mode that matters: a stale fetch must never leave the bot stuck in a blackout."""
  now = 1_700_000_000.0
  for calendar in (None, [], ["junk"], [{}], [{"name": "x"}], [{"name": "x", "ts": "bad"}],
                   [{"name": "x", "ts": float("nan")}]):
    assert macro_event_window(calendar, now) is None
    assert macro_event_entry_block(macro_event_window(calendar, now)) is None
  assert macro_event_window([_ev(30, now=now)], "not-a-time") is None


# --- the guard -----------------------------------------------------------------------------------

def test_guard_blocks_only_before_and_can_be_disabled():
  now = 1_700_000_000.0
  before = macro_event_window([_ev(20, now=now)], now)
  after = macro_event_window([_ev(-20, now=now)], now)
  assert "Macro event blackout" in macro_event_entry_block(before)
  assert "US CPI" in macro_event_entry_block(before)
  # Silent afterwards — that side belongs to the model as setup_family='macro_event'.
  assert macro_event_entry_block(after) is None
  assert macro_event_entry_block(before, enabled=False) is None


def test_macro_event_is_a_declarable_scored_family():
  """The tradeable half must go through the same scoreboard as every other playbook, so it
  explore-sizes while unproven and stands aside if it settles to no edge."""
  assert "macro_event" in SETUP_FAMILIES
  cfg = load_config().regime
  assert "macro_event" in cfg.declarable_setup_families
  assert cfg.macro_event_before_min == 60.0 and cfg.macro_event_after_min == 60.0


# --- the calendar store --------------------------------------------------------------------------

def test_calendar_stores_only_well_formed_future_events(tmp_path):
  m = MemoryStore(str(tmp_path / "m.json"))
  now = time.time()
  assert m.record_macro_events([
    {"name": "US CPI", "ts": now + 3600, "impact": "high"},
    {"name": "past", "ts": now - 60, "impact": "high"},        # already gone
    {"name": "", "ts": now + 60, "impact": "high"},            # no name
    {"name": "FOMC", "ts": "bad", "impact": "high"},           # unparseable
    "junk",
  ]) == 1
  assert [e["name"] for e in m.macro_events()] == ["US CPI"]
  assert m.macro_calendar_age_hours() == pytest.approx(0.0, abs=0.1)


def test_recording_replaces_rather_than_appends(tmp_path):
  """The tool sends the full forward list each refresh; appending would resurrect dropped events."""
  m = MemoryStore(str(tmp_path / "m.json"))
  now = time.time()
  m.record_macro_events([{"name": "A", "ts": now + 3600, "impact": "high"}])
  m.record_macro_events([{"name": "B", "ts": now + 7200, "impact": "high"}])
  assert [e["name"] for e in m.macro_events()] == ["B"]
  assert m.record_macro_events([]) == 0
  assert m.macro_events() == []          # cleared -> no blackout, never a stuck one


def test_past_events_are_swept_but_stay_available_for_the_after_window(tmp_path):
  m = MemoryStore(str(tmp_path / "m.json"))
  now = int(time.time())
  m.record_macro_events([{"name": "US CPI", "ts": now + 60, "impact": "high"}])
  data = m._read()
  data["macro_events"][0]["ts"] = now - 1800          # 30 min ago
  m._write(data)
  assert [e["name"] for e in m.macro_events()] == ["US CPI"]   # still visible for phase='after'
  data = m._read()
  data["macro_events"][0]["ts"] = now - 3 * 86400     # long gone
  m._write(m._prune(data))
  assert m.macro_events() == []
