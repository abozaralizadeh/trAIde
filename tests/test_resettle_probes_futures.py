"""The one-off re-settlement of stored probes on 1m FUTURES bars (scripts/resettle_probes_futures.py).

Fakes only: the script is meant to run once on the VM with the bot stopped; here it only ever sees a
temp file and an in-memory futures client.
"""
import json
import os
import time

import pytest

import scripts.resettle_probes_futures as R

T0 = 1_790_000_070          # a signal time (seconds), mid-minute like a real call


def _fut_row(ts_s, o, h, l, c):
  """A raw FUTURES kline row: [ts_ms, open, HIGH, LOW, close, volume, turnover]."""
  return [int(ts_s) * 1000, o, h, l, c, 10.0, 1.0]


class _Client:
  """Serves 1m futures bars from ``bars`` {ts_s: close} (o=h=l=c) and a funding history."""

  def __init__(self, closes, funding=(), *, bad_row=False, fail=()):
    self.closes = dict(closes)
    self.funding = list(funding)
    self.bad_row = bad_row
    self.fail = set(fail)
    self.calls = 0

  def get_candles(self, fsym, granularity=1, start_at=None, end_at=None):
    self.calls += 1
    if fsym in self.fail:
      raise RuntimeError("kline endpoint down")
    out = []
    for t, c in sorted(self.closes.items()):
      if start_at // 1000 <= t <= end_at // 1000:
        out.append(_fut_row(t, c, c, c, c))
    if self.bad_row and out:
      t, o, h, l, c = out[0][0] // 1000, 1.0, 1.0, 2.0, 1.5      # SPOT order read as futures
      out[0] = [t * 1000, o, h, l, c, 1.0, 1.0]
    return out

  def get_funding_rate_history(self, fsym, start_at=None, end_at=None):
    return [r for r in self.funding if start_at <= r["timepoint"] <= end_at]


def _minute_bars(start_s, minutes, price_fn):
  base = int(start_s) // 60 * 60
  return {base + 60 * i: price_fn(i) for i in range(minutes)}


def _memory(rows, exit_probes=()):
  return {"trades": [], "decisions": [], "signal_probes": rows, "exit_probes": list(exit_probes)}


def _probe(symbol, ts, base, side="long", family="funding_carry", stamps=None, source=None):
  ctx = {"positionSide": side, "marketPriceAtSignal": base, "setupFamily": family,
         "signalProbe": dict(stamps or {})}
  if source:
    ctx["priceSource"] = source
  return {"symbol": symbol, "ts": ts, "entryContext": ctx}


@pytest.fixture(autouse=True)
def _no_retry_sleep(monkeypatch):
  monkeypatch.setattr(R, "RETRY_PAUSE_SEC", 0)


# --- the bar checks ---------------------------------------------------------------------------------

def test_a_spot_ordered_row_is_refused_not_silently_misread():
  assert R.validate_bar(_fut_row(T0, 1.0, 2.0, 0.5, 1.5)) == [T0, 1.0, 2.0, 0.5, 1.5]
  with pytest.raises(R.ColumnOrderError):
    R.validate_bar([T0 * 1000, 1.0, 1.5, 2.0, 0.5])          # [ts, o, c, h, l] — the spot layout


def test_close_at_due_uses_the_bar_that_ended_at_the_due_minute_and_fills_at_most_three():
  bars = [[T0 - 40 + 60 * i, 0, 0, 0, 100.0 + i] for i in range(10)]   # starts on the minute
  starts = [b[0] for b in bars]
  due = bars[5][0] + 60                                       # the bar starting at [5] ends here
  assert R.close_at_due(bars, starts, due) == 105.0
  thin = [b for b in bars if b[0] < bars[3][0]]                # nothing traded after bar [2]
  thin_starts = [b[0] for b in thin]
  assert R.close_at_due(thin, thin_starts, bars[5][0]) == 102.0          # 2 min stale: carried
  assert R.close_at_due(thin, thin_starts, bars[5][0] + 60) == 102.0     # 3 min stale: carried
  assert R.close_at_due(thin, thin_starts, bars[5][0] + 120) is None     # 4 min: unmeasured


# --- the re-settlement ------------------------------------------------------------------------------

def test_spot_settled_rows_are_restamped_on_futures_with_funding_and_never_dropped():
  # ONE-USDT shape: base 0.0039 on the perp, spot stamps ~30% higher.
  bars = _minute_bars(T0 - 300, 300, lambda i: 0.0039 if i < 5 else 0.0040)   # moves after the call
  funding = [{"fundingRate": -0.001, "timepoint": (T0 + 1800) * 1000}]       # inside 60m, not 15m
  spot_stamps = {"m5": 0.0050, "m15": 0.0051, "m60": 0.0052, "m240": None}
  rows = [
    _probe("ONE-USDT", T0, 0.0039, stamps=spot_stamps),
    _probe("NEW-USDT", T0, 1.0, stamps={"m5": 1.0}, source="futures_mark"),   # settled live: untouched
  ]
  data = _memory(rows)
  report = R.resettle(data, _Client(bars, funding), now=T0 + 6 * 3600, sleep_sec=0, log=lambda _s: None)

  ctx = data["signal_probes"][0]["entryContext"]
  probe = ctx["signalProbe"]
  assert ctx["priceSource"] == R.RESETTLED_SOURCE
  assert probe["m5"] == 0.0040 and probe["m60"] == 0.0040 and probe["m240"] == 0.0040
  assert probe["f5"] == 0.0 and probe["f15"] == 0.0
  assert probe["f60"] == pytest.approx(0.001) and probe["f240"] == pytest.approx(0.001)  # a long is PAID
  assert data["signal_probes"][1]["entryContext"]["signalProbe"] == {"m5": 1.0}
  assert len(data["signal_probes"]) == 2                     # re-settled, never dropped
  assert report["rows"] == 1 and report["recovered"] == 1    # the m240 missed live is measured now
  assert report["barsChecked"] > 0


def test_a_horizon_with_no_bar_in_reach_is_unmeasured_not_borrowed():
  bars = _minute_bars(T0 - 300, 20, lambda i: 2.0)            # the contract stops trading after ~15 min
  data = _memory([_probe("THIN-USDT", T0, 2.0, stamps={"m60": 9.9})])
  R.resettle(data, _Client(bars), now=T0 + 6 * 3600, sleep_sec=0, log=lambda _s: None)
  probe = data["signal_probes"][0]["entryContext"]["signalProbe"]
  assert probe["m5"] == 2.0
  assert probe["m60"] is None and probe["m240"] is None


def test_a_symbol_that_cannot_be_fetched_is_left_exactly_as_it_was():
  row = _probe("DOWN-USDT", T0, 1.0, stamps={"m5": 1.3})
  data = _memory([json.loads(json.dumps(row))])
  report = R.resettle(data, _Client({}, fail={"DOWNUSDTM"}), now=T0 + 6 * 3600, sleep_sec=0,
                      log=lambda _s: None)
  assert data["signal_probes"][0] == row                     # still legacy, so a re-run retries it
  assert "DOWN-USDT" in report["failedSymbols"]


def test_a_misread_bar_aborts_the_whole_run():
  bars = _minute_bars(T0 - 300, 300, lambda i: 1.0)
  data = _memory([_probe("ONE-USDT", T0, 1.0)])
  with pytest.raises(R.ColumnOrderError):
    R.resettle(data, _Client(bars, bad_row=True), now=T0 + 6 * 3600, sleep_sec=0, log=lambda _s: None)


def test_an_exit_probe_resolved_after_its_life_becomes_unmeasured():
  late = {"symbol": "XMR-USDT", "ts": T0, "outcome": {"resolved": "stop", "bracketR": -1.0,
                                                     "resolvedTs": T0 + 7 * 86400}}
  timely = {"symbol": "XMR-USDT", "ts": T0, "outcome": {"resolved": "expired", "bracketR": 0.2,
                                                       "resolvedTs": T0 + 8 * 3600 + 60}}
  data = _memory([], exit_probes=[late, timely])
  assert R.mark_late_exit_probes(data) == 1
  assert data["exit_probes"][0]["outcome"]["resolved"] == "unmeasured"
  assert data["exit_probes"][0]["outcome"]["bracketR"] is None
  assert data["exit_probes"][0]["outcome"]["supersededOutcome"]["resolved"] == "stop"
  assert data["exit_probes"][1]["outcome"]["resolved"] == "expired"


# --- the command line: dry run by default, guarded apply ---------------------------------------------

def _file(tmp_path, data, age_sec=3600):
  path = tmp_path / "agent_memory.json"
  path.write_text(json.dumps(data), encoding="utf-8")
  old = time.time() - age_sec
  os.utime(path, (old, old))
  return path


def _client_for_now():
  now = time.time()
  ts = int(now) - 6 * 3600
  return ts, _Client(_minute_bars(ts - 300, 6 * 60 + 10, lambda i: 1.0 + i / 1000.0))


def test_dry_run_writes_nothing(tmp_path):
  ts, client = _client_for_now()
  path = _file(tmp_path, _memory([_probe("ONE-USDT", ts, 1.0, stamps={"m5": 5.0})]))
  before = path.read_bytes()
  assert R.main(["--memory", str(path), "--sleep", "0"], client=client, log=lambda _s: None) == 0
  assert path.read_bytes() == before
  assert not list(tmp_path.glob("*.bak"))


def test_apply_writes_and_keeps_a_backup(tmp_path):
  ts, client = _client_for_now()
  path = _file(tmp_path, _memory([_probe("ONE-USDT", ts, 1.0, stamps={"m5": 5.0})]))
  before = path.read_bytes()
  assert R.main(["--memory", str(path), "--apply", "--sleep", "0"], client=client, log=lambda _s: None) == 0
  saved = json.loads(path.read_text())
  assert saved["signal_probes"][0]["entryContext"]["priceSource"] == R.RESETTLED_SOURCE
  assert saved["signal_probes"][0]["entryContext"]["signalProbe"]["m5"] != 5.0
  backups = list(tmp_path.glob("agent_memory.json.pre-resettle-*.bak"))
  assert len(backups) == 1 and backups[0].read_bytes() == before


def test_apply_refuses_a_file_the_running_bot_just_wrote(tmp_path):
  ts, client = _client_for_now()
  path = _file(tmp_path, _memory([_probe("ONE-USDT", ts, 1.0)]), age_sec=10)
  before = path.read_bytes()
  assert R.main(["--memory", str(path), "--apply", "--sleep", "0"], client=client, log=lambda _s: None) == 3
  assert path.read_bytes() == before
  assert client.calls == 0                                   # refused before touching the network


# --- stackR backfill: agent closes scored against the live exit stack (2026-09-25) --------------------

def _legacy_trade(fill_ts, *, symbol="DASH-USDT", stop=98.0, family="range_edge"):
  """A filled LONG entered with 15m/1h bearish and 4h/1D bullish (the DASH/KCS shape)."""
  return {"symbol": symbol, "side": "buy", "filled": True, "fillTs": fill_ts, "fillPrice": 100.0,
          "ts": fill_ts - 30,
          "entryContext": {"positionSide": "long", "setupFamily": family, "entryPrice": 100.0,
                           "stopLossPrice": stop, "takeProfitPrice": 106.0, "stopAtrMult": 2.0,
                           "regime": {"intraday_bias_15m": "bearish", "intraday_bias_1h": "bearish",
                                      "intraday_bias_4h": "bullish", "daily_bias": "bullish"}}}


def _legacy_agent_probe(close_ts, *, symbol="DASH-USDT", stop=98.0):
  """An agent exit probe as recorded BEFORE the replay existed: no fillTs / initRiskPx / tags."""
  return {"symbol": symbol, "ts": close_ts, "positionSide": "long", "entryPrice": 100.0, "stopPrice": stop,
          "takeProfitPrice": 106.0, "exitPrice": 100.2, "realizedR": 0.1, "setupFamily": "funding_carry",
          "closedBy": "agent", "outcome": {"resolved": "take_profit", "bracketR": 3.0, "resolvedTs": close_ts + 3600}}


def _stack_cfg():
  from tests.test_protection import _cfg as _pp_cfg
  from src.protection import ProtectionManager
  return ProtectionManager(_pp_cfg(), None, breakeven_cost_pct=0.0032).cfg


def _trail_path(fill_ts):
  """Runs to +2R, retraces through the trail at +1.5R, then tags the +3R target (the INJ shape)."""
  base = (int(fill_ts) // 60 + 1) * 60
  closes = [100.4, 102.0, 104.0, 102.8] + [104.9, 106.2] + [106.2] * 600
  return base, closes


class _PathClient(_Client):
  """_Client with real high/low around each close for the first bars (the pullback bar dips to 102.5)."""

  def get_candles(self, fsym, granularity=1, start_at=None, end_at=None):
    rows = super().get_candles(fsym, granularity, start_at, end_at)
    out = []
    for r in rows:
      c = r[4]
      lo = 102.5 if c == 102.8 else c
      out.append([r[0], c, c, lo, c, 1.0, 1.0])
    return out


def test_backfill_scores_a_legacy_agent_close_on_the_live_stack_and_tags_it():
  close_ts = T0 + 120                                        # the model closed 2 minutes in
  fill = T0
  base, closes = _trail_path(fill)
  client = _PathClient({base + 60 * i: c for i, c in enumerate(closes)})
  data = {"trades": [_legacy_trade(fill)], "exit_probes": [_legacy_agent_probe(close_ts)]}
  report = R.backfill_exit_stacks(data, client, _stack_cfg(), now=close_ts + 9 * 3600, sleep_sec=0,
                                  log=lambda _s: None)
  row = data["exit_probes"][0]
  assert report["failed"] == {} and len(report["scored"]) == 1
  # Trail armed at +1R, ratcheted to 103 at +2R; the pullback bar OPENED through it at 102.8.
  assert row["stack"]["resolvedBy"] == "trail_stop" and row["stack"]["stackR"] == pytest.approx(1.4)
  assert row["stack"]["source"] == R.STACK_SOURCE
  assert row["outcome"]["bracketR"] == 3.0                  # the bracket stays for audit
  assert row["fillTs"] == fill and row["initRiskPx"] == pytest.approx(2.0)
  assert row["noiseBandR"] == pytest.approx(0.5)
  assert row["holdUntilTs"] is None                          # not a carry trade: no hold
  assert row["counterAtEntry"] is True and row["htfAligned"] is True
  from src.edge import exit_discipline_stats
  out = exit_discipline_stats(data["exit_probes"])
  assert out["stackScored"] == 1 and out["legacyBracketScored"] == 0

  # The same path on a funding_carry entry: the carry hold (first 8h-grid settlement after the fill,
  # the fallback for entries stamped before the contract's own clock was recorded) keeps the trail off,
  # so the stack rides to the target exactly as the live manager would have.
  carry = {"trades": [_legacy_trade(fill, family="funding_carry")], "exit_probes": [_legacy_agent_probe(close_ts)]}
  R.backfill_exit_stacks(carry, client, _stack_cfg(), now=close_ts + 9 * 3600, sleep_sec=0, log=lambda _s: None)
  row = carry["exit_probes"][0]
  assert row["holdUntilTs"] == (T0 // 28800 + 1) * 28800
  assert row["stack"]["resolvedBy"] == "take_profit" and row["stack"]["stackR"] == pytest.approx(3.0)


def test_backfill_leaves_rows_it_cannot_pin_down_untouched():
  close_ts = T0 + 1800
  probe = _legacy_agent_probe(close_ts, stop=97.0)            # no entry with this stop in the ledger
  young = _legacy_agent_probe(T0 + 9 * 3600, symbol="INJ-USDT")   # horizon not passed yet
  data = {"trades": [_legacy_trade(T0)], "exit_probes": [json.loads(json.dumps(probe)), young]}
  report = R.backfill_exit_stacks(data, _Client({}), _stack_cfg(), now=close_ts + 9 * 3600, sleep_sec=0,
                                  log=lambda _s: None)
  assert data["exit_probes"][0] == probe
  assert "stack" not in data["exit_probes"][1]
  assert any("entry not found" in v for v in report["failed"].values())


def test_backfill_aborts_on_a_misread_bar():
  close_ts = T0 + 1800
  base, closes = _trail_path(T0)
  data = {"trades": [_legacy_trade(T0)], "exit_probes": [_legacy_agent_probe(close_ts)]}
  with pytest.raises(R.ColumnOrderError):
    R.backfill_exit_stacks(data, _Client({base + 60 * i: c for i, c in enumerate(closes)}, bad_row=True),
                           _stack_cfg(), now=close_ts + 9 * 3600, sleep_sec=0, log=lambda _s: None)


def test_the_dry_run_reports_the_stack_backfill_and_writes_nothing(tmp_path):
  now = time.time()
  fill = int(now) - 12 * 3600
  base, closes = _trail_path(fill)
  data = _memory([], exit_probes=[_legacy_agent_probe(fill + 120)])
  data["trades"] = [_legacy_trade(fill)]
  path = _file(tmp_path, data)
  before = path.read_bytes()
  lines = []
  assert R.main(["--memory", str(path), "--sleep", "0"], client=_PathClient(
    {base + 60 * i: c for i, c in enumerate(closes)}), log=lines.append, stack_cfg=_stack_cfg()) == 0
  assert path.read_bytes() == before
  text = "\n".join(lines)
  assert "live exit stack" in text and "trail_stop" in text and "exitDiscipline after" in text


# --- the before/after report judges each SIDE, as the order path does (2026-09-25 review) -------------

def test_the_edge_summary_reports_the_stand_aside_per_side():
  """A pooled-open funding_carry whose shorts are refused on their own record must not print as open."""
  import random
  rng = random.Random(3)
  rows = []
  for i in range(46):
    side = "long" if i % 2 else "short"
    move = (0.03 if side == "long" else -0.004) + rng.gauss(0, 0.004)
    fwd = 1.0 * (1 + move) if side == "long" else 1.0 * (1 - move)
    rows.append(_probe(f"S{i}-USDT", T0 + i * 20_000, 1.0, side=side, stamps={f"m{h}": fwd for h in (5, 15, 60, 240)},
                       source="futures_mark"))
  out = R._edge_summary(_memory(rows), 0.0022)["families"]["funding_carry"]
  assert out["standAsideBySide"] == {"long": False, "short": True}
  assert out["standAside"] is False
  assert out["stakeBySide"]["short"] == 0.0 and out["stakeBySide"]["long"] > 0
  assert set(out["exploreBySide"]) == {"long", "short"}
  lines = []
  R._print_comparison({"families": {"funding_carry": out}, "by_horizon": {}},
                      {"families": {"funding_carry": out}, "by_horizon": {}}, lines.append)
  assert any("S:ASIDE" in ln and "L:" in ln for ln in lines)


# --- T8: the two safety paths of the full run ----------------------------------------------------------

def test_apply_refuses_when_the_file_changes_during_the_run(tmp_path):
  """The mtime/size re-check is the only guard left once --bot-stopped is passed: a running bot's poll
  rewriting the file mid-run must win, and nothing (not even a backup) is written."""
  ts, client = _client_for_now()
  path = _file(tmp_path, _memory([_probe("ONE-USDT", ts, 1.0, stamps={"m5": 5.0})]))
  bot_write = json.dumps(_memory([_probe("ONE-USDT", ts, 1.0, stamps={"m5": 7.0})]))
  orig_get = client.get_candles

  def get_candles(*a, **k):
    path.write_text(bot_write, encoding="utf-8")   # the running bot's poll rewrites the file
    return orig_get(*a, **k)

  client.get_candles = get_candles
  assert R.main(["--memory", str(path), "--apply", "--bot-stopped", "--sleep", "0"], client=client,
                log=lambda _s: None) == 4
  assert path.read_text(encoding="utf-8") == bot_write
  assert not list(tmp_path.glob("*.bak"))


def test_the_full_run_applies_the_late_exit_marking():
  bars = _minute_bars(T0 - 300, 300, lambda i: 0.0039)
  late = {"symbol": "XMR-USDT", "ts": T0, "outcome": {"resolved": "stop", "bracketR": -1.0,
                                                     "resolvedTs": T0 + 7 * 86400}}
  data = _memory([_probe("ONE-USDT", T0, 0.0039)], exit_probes=[late])
  report = R.resettle(data, _Client(bars), now=T0 + 9 * 3600, sleep_sec=0, log=lambda _s: None)
  assert report["exitProbesMarkedUnmeasured"] == 1
  assert data["exit_probes"][0]["outcome"]["resolved"] == "unmeasured"
  assert report["exitBrackets"]["reResolved"] == 0          # an unmeasured row is never re-resolved


# --- C3: the legacy entry survives the trades ledger's pruning (close records) --------------------------

def _close_decision(fill_ts, *, symbol="DASH-USDT", stop=98.0, pnl=0.05, family="range_edge"):
  """A realized close record as main logs it: the trade's entry context with fillTs/fillPrice."""
  ctx = dict(_legacy_trade(fill_ts, symbol=symbol, stop=stop, family=family)["entryContext"])
  ctx.update({"fillTs": fill_ts, "fillPrice": 100.0})
  return {"symbol": symbol, "action": "futures_close_long", "pnl": pnl, "ts": fill_ts + 200,
          "positionSide": "long", "entryContext": ctx}


def test_backfill_finds_a_pruned_entry_in_the_close_record():
  close_ts = T0 + 120
  base, closes = _trail_path(T0)
  client = _PathClient({base + 60 * i: c for i, c in enumerate(closes)})
  data = {"trades": [], "decisions": [_close_decision(T0)], "exit_probes": [_legacy_agent_probe(close_ts)]}
  report = R.backfill_exit_stacks(data, client, _stack_cfg(), now=close_ts + 9 * 3600, sleep_sec=0,
                                  log=lambda _s: None)
  row = data["exit_probes"][0]
  assert report["failed"] == {} and len(report["scored"]) == 1
  assert row["fillTs"] == T0 and row["initRiskPx"] == pytest.approx(2.0) and row["noiseBandR"] == pytest.approx(0.5)
  assert row["stack"]["resolvedBy"] == "trail_stop" and row["stack"]["stackR"] == pytest.approx(1.4)


def test_a_close_record_with_another_stop_or_a_later_fill_is_not_matched():
  close_ts = T0 + 120
  for dec in (_close_decision(T0, stop=97.0),                  # a different original stop
              _close_decision(close_ts + 60),                   # filled after the close
              {**_close_decision(T0), "pnl": None},             # not a realized close
              _close_decision(T0, symbol="INJ-USDT")):          # another symbol
    probe = _legacy_agent_probe(close_ts)
    data = {"trades": [], "decisions": [dec], "exit_probes": [json.loads(json.dumps(probe))]}
    report = R.backfill_exit_stacks(data, _Client({}), _stack_cfg(), now=close_ts + 9 * 3600, sleep_sec=0,
                                    log=lambda _s: None)
    assert data["exit_probes"][0] == probe
    assert any("entry not found in the trades ledger or close records" in v for v in report["failed"].values())


# --- C1: spot-settled exit brackets are re-resolved on futures closes ----------------------------------

def _exit_probe(symbol, ts, *, resolved="take_profit", bracket=1.7, source=None, who="protection"):
  row = {"symbol": symbol, "ts": ts, "positionSide": "long", "entryPrice": 100.0, "stopPrice": 98.0,
         "takeProfitPrice": 103.4, "exitPrice": 100.5, "realizedR": 0.25, "closedBy": who,
         "outcome": {"resolved": resolved, "bracketR": bracket, "resolvedTs": ts + 60}}
  if source:
    row["outcome"]["priceSource"] = source
  return row


def test_a_spot_phantom_take_profit_is_re_resolved_to_the_futures_stop():
  """ONE-USDT 09-21 00:04 shape: stored as a +1.70R TP one minute later on spot; on the contract it
  was stopped. An already futures-stamped row and a row without bars are left exactly as they were."""
  bars = _minute_bars(T0, 9 * 60, lambda i: 99.5 if i < 30 else 97.5)       # drifts down, through 98
  phantom = _exit_probe("ONE-USDT", T0)
  stamped = _exit_probe("ONE-USDT", T0 + 60, source="futures_mark")
  missing = _exit_probe("DOWN-USDT", T0)
  data = _memory([], exit_probes=[phantom, json.loads(json.dumps(stamped)), json.loads(json.dumps(missing))])
  client = _Client(bars, fail={"DOWNUSDTM"})
  report = R.resettle_exit_brackets(data, client, now=T0 + 10 * 3600, sleep_sec=0, log=lambda _s: None)
  out = data["exit_probes"][0]["outcome"]
  assert out["resolved"] == "stop" and out["bracketR"] == -1.0
  assert out["priceSource"] == R.EXIT_BRACKET_SOURCE
  assert out["supersededOutcome"]["resolved"] == "take_profit" and out["supersededOutcome"]["bracketR"] == 1.7
  assert out["resolvedTs"] == (T0 // 60 * 60) + 31 * 60                      # the close that crossed 98
  assert data["exit_probes"][1] == stamped                                    # live-stamped: never touched
  assert data["exit_probes"][2] == missing                                    # no bars: untouched, listed
  assert report["reResolved"] == 1 and report["flipped"][0]["to"] == "stop"
  assert any(k.startswith("DOWN-USDT") for k in report["failed"])
  # Re-run safe: the re-resolved row now carries a priceSource.
  again = R.resettle_exit_brackets(data, client, now=T0 + 10 * 3600, sleep_sec=0, log=lambda _s: None)
  assert again["reResolved"] == 0


def test_bracket_resolution_is_on_closes_and_an_incomplete_window_is_refused():
  bars = [R.validate_bar(_fut_row(T0 + 60 * i, 100.0, 103.6, 99.0, 100.2)) for i in range(8 * 60 + 5)]
  got = R.resolve_bracket_on_closes(bars, side_long=True, entry=100.0, stop=98.0, take_profit=103.4,
                                    start_ts=T0, end_ts=T0 + 8 * 3600)
  assert got[0] == "expired" and got[1] == pytest.approx(0.1)               # the wicks never count
  short = R.resolve_bracket_on_closes(bars[:60], side_long=True, entry=100.0, stop=98.0, take_profit=103.4,
                                      start_ts=T0, end_ts=T0 + 8 * 3600)
  assert short is None
