from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import product
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from .analytics import candles_to_dataframe, compute_indicators
from .config import load_config
from .kucoin import KucoinClient


@dataclass
class TradeResult:
  entry_idx: int
  exit_idx: int
  entry_price: float
  exit_price: float
  reason: str
  pnl_pct: float
  win: bool


@dataclass
class BacktestResult:
  trades: List[TradeResult]
  total_return_pct: float
  win_rate: float
  profit_factor: float
  max_drawdown_pct: float


def _simulate(df: pd.DataFrame, params: Dict[str, float]) -> BacktestResult:
  df = compute_indicators(df)
  trades: List[TradeResult] = []
  position = None
  entry_idx = -1
  peak_equity = 1.0
  equity = 1.0
  fee = params.get("fee", 0.001)
  stop_mult = params.get("stop_atr_mult", 1.5)
  target_mult = params.get("target_atr_mult", 2.0)
  buy_rsi = params.get("buy_rsi", 55)
  min_macd_hist = params.get("min_macd_hist", 0)

  for idx, row in df.iterrows():
    price = float(row["close"])
    atr = float(row["atr"]) if not pd.isna(row["atr"]) else None
    if atr is None:
      continue

    if position is None:
      bullish = row["ema_fast"] > row["ema_slow"] and row["rsi"] >= buy_rsi and row["macd_hist"] >= min_macd_hist
      if bullish:
        position = {
          "entry": price,
          "stop": max(0.0, price - atr * stop_mult),
          "target": price + atr * target_mult,
        }
        entry_idx = idx
    else:
      hit_stop = row["low"] <= position["stop"]
      hit_target = row["high"] >= position["target"]
      exit_reason = None
      exit_price = price
      if hit_stop and hit_target:
        # Favor target first when both touched; adjust if desired
        exit_reason = "both-hit-target-priority"
        exit_price = position["target"]
      elif hit_target:
        exit_reason = "target"
        exit_price = position["target"]
      elif hit_stop:
        exit_reason = "stop"
        exit_price = position["stop"]
      elif row["ema_fast"] < row["ema_slow"]:
        exit_reason = "trend_flip"
        exit_price = price

      if exit_reason:
        gross_pct = (exit_price - position["entry"]) / position["entry"] * 100
        net_pct = gross_pct - fee * 200  # entry + exit fees in bps terms
        win = net_pct > 0
        equity *= 1 + net_pct / 100
        peak_equity = max(peak_equity, equity)
        drawdown_pct = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0
        trades.append(
          TradeResult(
            entry_idx=entry_idx,
            exit_idx=idx,
            entry_price=position["entry"],
            exit_price=exit_price,
            reason=exit_reason,
            pnl_pct=net_pct,
            win=win,
          )
        )
        position = None

  wins = [t for t in trades if t.win]
  losses = [t for t in trades if not t.win]
  win_rate = len(wins) / len(trades) * 100 if trades else 0.0
  gross_gain = sum(max(t.pnl_pct, 0) for t in trades)
  gross_loss = abs(sum(min(t.pnl_pct, 0) for t in trades))
  profit_factor = (gross_gain / gross_loss) if gross_loss > 0 else float("inf") if gross_gain > 0 else 0.0
  total_return_pct = (equity - 1) * 100
  max_drawdown_pct = 0.0
  if trades:
    running_equity = 1.0
    peak = 1.0
    for t in trades:
      running_equity *= 1 + t.pnl_pct / 100
      peak = max(peak, running_equity)
      dd = (peak - running_equity) / peak * 100 if peak > 0 else 0
      max_drawdown_pct = max(max_drawdown_pct, dd)

  return BacktestResult(
    trades=trades,
    total_return_pct=total_return_pct,
    win_rate=win_rate,
    profit_factor=profit_factor,
    max_drawdown_pct=max_drawdown_pct,
  )


def run_backtest(symbol: str, interval: str = "1hour", lookback_hours: int = 240, **params: float) -> BacktestResult:
  cfg = load_config()
  client = KucoinClient(cfg)
  points = min(500, max(50, int(lookback_hours * 3600 / 3600)))  # coarse bound
  data = client.get_candles(symbol, interval=interval, start_at=None, end_at=None)
  if not data:
    raise RuntimeError("No candles returned for backtest.")
  df = candles_to_dataframe(data[-points:])
  return _simulate(df, params)


def run_sweep(
  symbol: str,
  interval: str,
  lookback_hours: int,
  buy_rsi: Sequence[float],
  stop_atr_mult: Sequence[float],
  target_atr_mult: Sequence[float],
  min_macd_hist: Sequence[float],
  fee: float,
) -> List[Tuple[Dict[str, float], BacktestResult]]:
  cfg = load_config()
  client = KucoinClient(cfg)
  points = min(500, max(50, int(lookback_hours * 3600 / 3600)))
  data = client.get_candles(symbol, interval=interval, start_at=None, end_at=None)
  if not data:
    raise RuntimeError("No candles returned for sweep.")
  df = candles_to_dataframe(data[-points:])

  results: List[Tuple[Dict[str, float], BacktestResult]] = []
  for brsi, stopm, targm, macd in product(buy_rsi, stop_atr_mult, target_atr_mult, min_macd_hist):
    params = {
      "buy_rsi": brsi,
      "stop_atr_mult": stopm,
      "target_atr_mult": targm,
      "min_macd_hist": macd,
      "fee": fee,
    }
    res = _simulate(df, params)
    results.append((params, res))
  # sort by total return then max drawdown
  results.sort(key=lambda item: (item[1].total_return_pct, -item[1].max_drawdown_pct), reverse=True)
  return results


def _cli() -> None:
  parser = argparse.ArgumentParser(description="Lightweight EMA/RSI/ATR backtest harness.")
  parser.add_argument("--symbol", required=True, help="Symbol e.g. BTC-USDT")
  parser.add_argument("--interval", default="1hour", help="Candle interval (default: 1hour)")
  parser.add_argument("--lookback_hours", type=int, default=240)
  parser.add_argument("--buy_rsi", type=float, default=55)
  parser.add_argument("--stop_atr_mult", type=float, default=1.5)
  parser.add_argument("--target_atr_mult", type=float, default=2.0)
  parser.add_argument("--min_macd_hist", type=float, default=0.0)
  parser.add_argument("--fee", type=float, default=0.001, help="Per-side fee in decimal (0.1% default).")
  parser.add_argument("--sweep", action="store_true", help="Run parameter sweep across ranges.")
  parser.add_argument("--buy_rsi_range", default="", help="Comma list e.g. 50,55,60")
  parser.add_argument("--stop_mult_range", default="", help="Comma list e.g. 1.0,1.5,2.0")
  parser.add_argument("--target_mult_range", default="", help="Comma list e.g. 1.5,2.0,2.5")
  parser.add_argument("--macd_hist_range", default="", help="Comma list e.g. -0.1,0,0.1")
  args = parser.parse_args()

  if args.sweep:
    buy_rsi_vals = [float(x) for x in args.buy_rsi_range.split(",") if x.strip()] or [args.buy_rsi]
    stop_vals = [float(x) for x in args.stop_mult_range.split(",") if x.strip()] or [args.stop_atr_mult]
    target_vals = [float(x) for x in args.target_mult_range.split(",") if x.strip()] or [args.target_atr_mult]
    macd_vals = [float(x) for x in args.macd_hist_range.split(",") if x.strip()] or [args.min_macd_hist]
    results = run_sweep(
      args.symbol,
      interval=args.interval,
      lookback_hours=args.lookback_hours,
      buy_rsi=buy_rsi_vals,
      stop_atr_mult=stop_vals,
      target_atr_mult=target_vals,
      min_macd_hist=macd_vals,
      fee=args.fee,
    )
    print(f"Sweep results (top 5 of {len(results)} combos):")
    for params, res in results[:5]:
      print(
        f"RSI {params['buy_rsi']} stop {params['stop_atr_mult']} target {params['target_atr_mult']} macd {params['min_macd_hist']} "
        f"trades {len(res.trades)} return {res.total_return_pct:.2f}% win {res.win_rate:.1f}% pf {res.profit_factor:.2f} dd {res.max_drawdown_pct:.2f}%"
      )
  else:
    res = run_backtest(
      args.symbol,
      interval=args.interval,
      lookback_hours=args.lookback_hours,
      buy_rsi=args.buy_rsi,
      stop_atr_mult=args.stop_atr_mult,
      target_atr_mult=args.target_atr_mult,
      min_macd_hist=args.min_macd_hist,
      fee=args.fee,
    )

    print(f"Trades: {len(res.trades)}")
    print(f"Total Return: {res.total_return_pct:.2f}%")
    print(f"Win Rate: {res.win_rate:.1f}%")
    print(f"Profit Factor: {res.profit_factor:.2f}")
    print(f"Max Drawdown: {res.max_drawdown_pct:.2f}%")


if __name__ == "__main__":
  _cli()
