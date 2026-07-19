import pandas as pd
import pytest

from src import backtest


def test_simulate_resolves_same_candle_target_and_stop_at_stop(monkeypatch):
    enriched = pd.DataFrame([
        {
            "close": 100.0,
            "high": 101.0,
            "low": 99.0,
            "atr": 10.0,
            "ema_fast": 2.0,
            "ema_slow": 1.0,
            "rsi": 60.0,
            "macd_hist": 1.0,
        },
        {
            "close": 100.0,
            "high": 125.0,
            "low": 80.0,
            "atr": 10.0,
            "ema_fast": 2.0,
            "ema_slow": 1.0,
            "rsi": 60.0,
            "macd_hist": 1.0,
        },
    ])
    monkeypatch.setattr(backtest, "compute_indicators", lambda df: df)

    result = backtest._simulate(
        enriched,
        {"fee": 0.0, "stop_atr_mult": 1.5, "target_atr_mult": 2.0},
    )

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.reason == "both-hit-stop-priority"
    assert trade.exit_price == pytest.approx(85.0)
    assert trade.pnl_pct == pytest.approx(-15.0)
    assert trade.win is False


@pytest.mark.parametrize(
    ("interval", "lookback_hours", "expected"),
    [
        ("1hour", 100, 100),
        ("15min", 100, 400),
        ("4hour", 240, 60),
    ],
)
def test_lookback_points_use_requested_interval(interval, lookback_hours, expected):
    assert backtest._lookback_points(interval, lookback_hours) == expected


def test_lookback_points_reject_unknown_interval():
    with pytest.raises(ValueError, match="Unsupported candle interval"):
        backtest._lookback_points("not-an-interval", 100)


def test_prepare_frame_sorts_newest_first_and_excludes_open_bar():
    def candle(ts, close):
        return [ts, close, close, close + 1, close - 1, 1, 1]

    # Exchange order is newest-first. At t=10_800, the 7_200 bar has just closed and 10_800 is open.
    rows = [candle(10_800, 4), candle(7_200, 3), candle(3_600, 2), candle(0, 1)]
    frame = backtest._prepare_backtest_frame(rows, "1hour", 2, as_of=10_800)
    assert frame["time"].tolist() == [3_600, 7_200]
    assert frame["close"].tolist() == [2, 3]
