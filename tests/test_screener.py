"""Tests for the market screener (_screen_contracts in src/agent.py).

Gives the research scout eyes on the whole perp universe instead of only symbols it already names.
"""

from types import SimpleNamespace

from src.agent import _screen_contracts

NOW = 2_000_000_000  # fixed 'now' (seconds) for deterministic age math


def _c(sym, chg, turnover, age_days=30, status="Open", funding=0.0001):
    first_open_ms = (NOW - age_days * 86400) * 1000
    return {
        "symbol": sym,
        "priceChgPct": chg,           # decimal fraction (0.05 = +5%)
        "turnoverOf24h": turnover,    # USDT
        "firstOpenDate": first_open_ms,
        "status": status,
        "fundingFeeRate": funding,
        "markPrice": 100.0,
    }


def _universe():
    return [
        _c("XBTUSDTM", 0.01, 500_000_000),   # BTC: liquid, small move
        _c("ETHUSDTM", 0.03, 200_000_000),   # ETH: liquid, modest move
        _c("DOGEUSDTM", 0.12, 40_000_000),   # big gainer, liquid
        _c("WILDUSDTM", -0.18, 20_000_000),  # big loser, liquid
        _c("THINUSDTM", 0.40, 100_000),      # huge move but ILLIQUID -> filtered
        _c("FRESHUSDTM", 0.30, 50_000_000, age_days=2),  # liquid but TOO NEW -> filtered
        _c("DEADUSDTM", 0.25, 30_000_000, status="Paused"),  # not Open -> filtered
        {"symbol": "ETH-USDT", "priceChgPct": 0.5, "turnoverOf24h": 9e9},  # spot-form, not *USDTM -> skipped
    ]


def _screen(**kw):
    base = dict(min_turnover=5_000_000, min_age_days=7, now=NOW, sort_by="momentum", side="both", top_n=15)
    base.update(kw)
    return _screen_contracts(_universe(), **base)


def test_liquidity_and_age_and_status_filters():
    res = _screen()
    syms = {r["futuresSymbol"] for r in res["results"]}
    assert "THINUSDTM" not in syms   # illiquid
    assert "FRESHUSDTM" not in syms  # too new
    assert "DEADUSDTM" not in syms   # not Open
    # 4 qualify: BTC, ETH, DOGE, WILD
    assert res["qualified"] == 4


def test_momentum_sort_puts_biggest_absolute_move_first():
    res = _screen(sort_by="momentum")
    # WILD -18% has the biggest absolute move among the qualified
    assert res["results"][0]["futuresSymbol"] == "WILDUSDTM"


def test_gainers_and_losers_and_volume_sorts():
    assert _screen(sort_by="gainers")["results"][0]["futuresSymbol"] == "DOGEUSDTM"
    assert _screen(sort_by="losers")["results"][0]["futuresSymbol"] == "WILDUSDTM"
    assert _screen(sort_by="volume")["results"][0]["futuresSymbol"] == "XBTUSDTM"


def test_side_filter():
    longs = _screen(side="long")["results"]
    assert all((r["chgPct24h"] or 0) > 0 for r in longs)
    assert "WILDUSDTM" not in {r["futuresSymbol"] for r in longs}
    shorts = _screen(side="short")["results"]
    assert all((r["chgPct24h"] or 0) < 0 for r in shorts)
    assert {r["futuresSymbol"] for r in shorts} == {"WILDUSDTM"}


def test_symbol_normalized_to_spot_form():
    res = _screen(sort_by="volume")
    top = res["results"][0]
    assert top["futuresSymbol"] == "XBTUSDTM" and top["symbol"] == "BTC-USDT"  # XBT->BTC
    assert top["chgPct24h"] == 1.0  # 0.01 -> 1.0%


def test_top_n_capped():
    assert len(_screen(top_n=2)["results"]) == 2
    assert len(_screen(top_n=999)["results"]) == 4  # only 4 qualify


def test_turnover_floor_can_be_disabled():
    # min_turnover=0 lets the illiquid mover through
    res = _screen(min_turnover=0, sort_by="gainers")
    assert res["results"][0]["futuresSymbol"] == "THINUSDTM"


def test_extreme_24h_movers_can_be_excluded_before_ranking():
    # WILD is the top momentum name but its -18% move is beyond a 15% execution cap.
    res = _screen(max_abs_change_pct=15)
    syms = {r["futuresSymbol"] for r in res["results"]}
    assert "WILDUSDTM" not in syms
    assert res["results"][0]["futuresSymbol"] == "DOGEUSDTM"


def test_empty_universe():
    res = _screen_contracts([], min_turnover=5_000_000, min_age_days=7, now=NOW)
    assert res["qualified"] == 0 and res["results"] == []


# ── Execution reachability: contracts this bot cannot trade never take a slot (2026-09-25) ──────────

from src.agent import NOT_TRADEABLE_HERE, _bot_can_trade  # noqa: E402

# Spot pairs that exist and trade. FHE and TAKE are perp-only; 1000BONK's spot pair is BONK-USDT.
_SPOT = {"BTC-USDT", "ETH-USDT", "DOGE-USDT", "WILD-USDT", "BONK-USDT"}


def _reach_universe():
  return [
    _c("XBTUSDTM", 0.01, 500_000_000),
    _c("ETHUSDTM", 0.03, 200_000_000),
    _c("FHEUSDTM", -0.25, 30_000_000),     # biggest mover, liquid, NO spot pair
    _c("TAKEUSDTM", -0.20, 12_000_000),    # perp-only too
    _c("DOGEUSDTM", 0.12, 40_000_000),
    _c("WILDUSDTM", -0.18, 20_000_000),
    _c("1000BONKUSDTM", 0.15, 25_000_000),  # spot is BONK-USDT: the round trip must reject it
  ]


def _reach(**kw):
  base = dict(min_turnover=5_000_000, min_age_days=7, now=NOW, sort_by="momentum", side="both", top_n=3,
              tradeable=lambda fsym: _bot_can_trade(fsym, _SPOT))
  base.update(kw)
  return _screen_contracts(_reach_universe(), **base)


class TestExecutionReachability:
  def test_bot_can_trade_requires_a_live_spot_pair_and_the_round_trip(self):
    assert _bot_can_trade("XBTUSDTM", _SPOT) is True        # BTC <-> XBT maps both ways
    assert _bot_can_trade("DOGE-USDT", _SPOT) is True
    assert _bot_can_trade("FHEUSDTM", _SPOT) is False       # perp-only
    assert _bot_can_trade("1000BONKUSDTM", _SPOT) is False  # BONK-USDT exists, but not 1000BONK-USDT
    # The round trip: 'BTCUSDTM' normalizes to the listed BTC-USDT, but the execution layer would trade
    # XBTUSDTM for it, never BTCUSDTM — so that name is not reachable.
    assert _bot_can_trade("BTCUSDTM", _SPOT) is False
    assert _bot_can_trade("FHEUSDTM", None) is None         # list unknown -> caller fails open

  def test_untradeable_perps_are_excluded_before_the_top_n_cut(self):
    res = _reach(top_n=3)
    syms = [r["futuresSymbol"] for r in res["results"]]
    # Without the predicate FHE and TAKE would hold two of the three slots.
    assert syms == ["WILDUSDTM", "DOGEUSDTM", "ETHUSDTM"]
    assert res["excluded"] == {"FHE-USDT": NOT_TRADEABLE_HERE, "TAKE-USDT": NOT_TRADEABLE_HERE,
                               "1000BONK-USDT": NOT_TRADEABLE_HERE}
    assert res["tradeableCheck"] == "applied"
    assert res["qualified"] == 7                          # the universe count is unchanged

  def test_excluded_names_only_what_would_have_taken_a_slot(self):
    # top_n=1: WILD is the first tradeable row; only the untradeable names ranked above it are named.
    res = _reach(top_n=1)
    assert [r["futuresSymbol"] for r in res["results"]] == ["WILDUSDTM"]
    assert set(res["excluded"]) == {"FHE-USDT", "TAKE-USDT"}

  def test_spot_list_unavailable_fails_open_and_leaves_rows_unmarked(self):
    res = _reach(top_n=3, tradeable=lambda fsym: None)
    syms = [r["futuresSymbol"] for r in res["results"]]
    assert syms[0] == "FHEUSDTM"                          # nothing dropped
    assert res["excluded"] == {} and res["tradeableCheck"] == "unavailable"
    assert all("tradeable" not in r for r in res["results"])

  def test_a_predicate_that_raises_keeps_the_row(self):
    def boom(fsym):
      raise RuntimeError("lookup broke")
    res = _reach(top_n=2, tradeable=boom)
    assert [r["futuresSymbol"] for r in res["results"]] == ["FHEUSDTM", "TAKEUSDTM"]
    assert res["tradeableCheck"] == "unavailable"

  def test_no_predicate_keeps_the_old_shape(self):
    res = _screen()
    assert "excluded" not in res and "tradeableCheck" not in res

  def test_rows_carry_the_contracts_asset_class(self):
    universe = [dict(_c("TSLAUSDTM", 0.05, 50_000_000), assetClass="STOCK"),
                dict(_c("DOGEUSDTM", 0.12, 40_000_000), assetClass="CRYPTO")]
    res = _screen_contracts(universe, min_turnover=5_000_000, min_age_days=7, now=NOW)
    assert {r["futuresSymbol"]: r["assetClass"] for r in res["results"]} == {
      "TSLAUSDTM": "STOCK", "DOGEUSDTM": "CRYPTO"}


# ── scan_futures_market and add_coin share ONE reachability check (call sites, on fakes) ───────────


class _ScanFutures:
  def __init__(self, contracts):
    self.contracts = contracts

  def list_active_contracts(self):
    return self.contracts


class _SpotList:
  """Spot client fake: the live symbol list, and a ticker that exists only for listed pairs."""

  def __init__(self, symbols=_SPOT, *, fail=False):
    self.symbols, self.fail, self.list_calls = set(symbols), fail, 0

  def list_symbols(self):
    self.list_calls += 1
    if self.fail:
      raise RuntimeError("symbols endpoint down")
    return [{"symbol": s, "enableTrading": True} for s in sorted(self.symbols)] + [
      {"symbol": "HALT-USDT", "enableTrading": False}]

  def get_ticker(self, symbol):
    if symbol not in self.symbols:
      raise RuntimeError("Kucoin ticker unavailable")
    return SimpleNamespace(price=1.0)


def _scan_tools(tmp_path, *, spot=None, contracts=None, memory=None):
  from src.config import load_config
  from src.memory import MemoryStore
  from src.tools import build_tools

  cfg = load_config()
  cfg.kucoin_futures.enabled = True
  cfg.trading.screener_min_turnover_usd_24h = 5_000_000
  cfg.trading.min_futures_listing_age_days = 7
  cfg.trading.max_24h_volatility_pct = 0.0
  memory = memory or MemoryStore(str(tmp_path / "scan.json"))
  snapshot = SimpleNamespace(tickers={}, coins=[], futures_positions=[], futures_pending_orders=[],
                             paper_trading=True)
  ctx = SimpleNamespace(
    cfg=cfg, kucoin=spot or _SpotList(), kucoin_futures=_ScanFutures(contracts or _now_universe()),
    memory=memory, snapshot=snapshot, allowed_symbols={"ETH-USDT"}, balances_by_currency={},
    fees={"futures_taker": 0.0006}, _daily_gate_state={}, _futures_margin_mode="cross",
    _apply_cross_leverage=lambda *a, **k: None, _btc_daily_bias=lambda: "neutral",
    _edge_state=lambda: {}, _fee_adjusted_breakeven=lambda *a, **k: 0.0,
    _get_contract_spec=lambda fsym: None, _repair_allowed_symbol=lambda s: None,
    _spot_position_info=lambda *a, **k: None, _spot_position_size=lambda *a, **k: 0.0,
    _stop_distance_ok=lambda *a, **k: (True, None), safety_state=None, entry_token=None,
  )
  return build_tools(ctx), memory


def _now_universe():
  import time as _t
  now = _t.time()
  first_open = (now - 30 * 86400) * 1000
  return [dict(c, firstOpenDate=first_open) for c in _reach_universe()]


def _invoke(tool, **args):
  import asyncio
  import json
  from agents.tool_context import ToolContext
  raw = json.dumps(args)
  return asyncio.run(tool.on_invoke_tool(
    ToolContext(context=None, tool_name=tool.name, tool_call_id="t1", tool_arguments=raw), raw))


class TestScanAndAddCoinShareTheCheck:
  def test_scan_excludes_perp_only_rows_through_the_live_spot_list(self, tmp_path):
    tools, _ = _scan_tools(tmp_path)
    out = _invoke(tools.scan_futures_market, top_n=3)
    assert [r["futuresSymbol"] for r in out["results"]] == ["WILDUSDTM", "DOGEUSDTM", "ETHUSDTM"]
    assert set(out["excluded"]) == {"FHE-USDT", "TAKE-USDT", "1000BONK-USDT"}
    assert out["tradeableCheck"] == "applied"

  def test_scan_fails_open_when_the_spot_list_is_down(self, tmp_path):
    tools, _ = _scan_tools(tmp_path, spot=_SpotList(fail=True))
    out = _invoke(tools.scan_futures_market, top_n=3)
    assert out["results"][0]["futuresSymbol"] == "FHEUSDTM"
    assert out["excluded"] == {} and out["tradeableCheck"] == "unavailable"

  def test_add_coin_refuses_a_perp_only_name_with_the_same_reason(self, tmp_path):
    tools, memory = _scan_tools(tmp_path)
    out = _invoke(tools.add_coin, symbol="FHE-USDT", reason="momentum")
    assert out.get("rejected") is True and NOT_TRADEABLE_HERE in out["reason"]
    assert "FHE-USDT" not in memory.get_coins()

  def test_add_coin_and_scan_call_the_same_helper(self):
    import inspect
    import src.tools as tools_mod
    src = inspect.getsource(tools_mod.build_tools)
    add = src[src.index("async def add_coin("):src.index("async def remove_coin(")]
    scan = src[src.index("async def scan_futures_market("):src.index("async def fetch_futures_orderbook(")]
    assert "_tradeable_by_bot(norm) is False" in add
    assert "tradeable=_tradeable_by_bot" in scan
    helper = src[src.index("def _tradeable_by_bot("):src.index("@function_tool")]
    assert "_bot_can_trade(symbol, _SPOT_SYMBOLS.get(kucoin))" in helper

  def test_spot_list_is_cached_per_client(self, tmp_path):
    from src.tools import _SpotSymbolCache
    cache = _SpotSymbolCache()
    a, b = _SpotList(), _SpotList(symbols={"ETH-USDT"})
    assert "HALT-USDT" not in cache.get(a, now=1000.0)           # enableTrading False is not tradeable
    assert cache.get(a, now=2000.0) is cache.get(a, now=2500.0) and a.list_calls == 1
    assert cache.get(b, now=2600.0) == {"ETH-USDT"} and b.list_calls == 1  # another client never reads a's
    assert cache.get(a, now=1000.0 + _SpotSymbolCache.TTL_SEC + 1) is not None and a.list_calls == 2

  def test_an_outage_is_asked_once_per_retry_window_not_once_per_row(self, tmp_path):
    spot = _SpotList(fail=True)
    tools, _ = _scan_tools(tmp_path, spot=spot)
    _invoke(tools.scan_futures_market, top_n=5)
    _invoke(tools.add_coin, symbol="FHE-USDT", reason="momentum")
    assert spot.list_calls == 1

  def test_quarantine_and_analysis_failure_annotate_the_row(self, tmp_path):
    import time as _t
    from src.memory import MemoryStore
    memory = MemoryStore(str(tmp_path / "ann.json"))
    memory.remove_coin("WILD-USDT", reason="Automatic risk quarantine: daily ATR 30.00% exceeds 12.00% hard limit",
                       exit_plan="x")
    memory.record_analysis_failure("DOGE-USDT", reason="1hour: 2 candle gap(s)", retry_after=_t.time() + 7200)
    tools, _ = _scan_tools(tmp_path, memory=memory)
    out = _invoke(tools.scan_futures_market, top_n=3)
    rows = {r["futuresSymbol"]: r for r in out["results"]}
    assert rows["WILDUSDTM"]["quarantined"] is True and rows["WILDUSDTM"]["remainingHours"] > 0
    assert rows["DOGEUSDTM"]["lastAnalysisFailure"] == "1hour: 2 candle gap(s)"
    assert 1.8 <= rows["DOGEUSDTM"]["retryInHours"] <= 2.0 and rows["DOGEUSDTM"]["retryAfter"] > _t.time()
    assert "quarantined" not in rows["ETHUSDTM"] and "lastAnalysisFailure" not in rows["ETHUSDTM"]
    # Annotated, never dropped.
    assert list(rows) == ["WILDUSDTM", "DOGEUSDTM", "ETHUSDTM"]
