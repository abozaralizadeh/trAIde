"""Tests for the market screener (_screen_contracts in src/agent.py).

Gives the research scout eyes on the whole perp universe instead of only symbols it already names.
"""

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
