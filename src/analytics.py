from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import pandas as pd

# Kucoin candle fields: [time, open, close, high, low, volume, turnover]
INTERVAL_SECONDS: Dict[str, int] = {"1min": 60, "5min": 300, "15min": 900, "1hour": 3600}


@dataclass
class IndicatorSettings:
  ema_fast: int = 12
  ema_slow: int = 26
  ema_signal: int = 9
  rsi_length: int = 14
  atr_length: int = 14
  bb_length: int = 20
  bb_stddev: float = 2.0


def candles_to_dataframe(candles: Sequence[Sequence[Any]]) -> pd.DataFrame:
  if not candles:
    raise ValueError("No candle data to analyze.")
  df = pd.DataFrame(
    candles,
    columns=["time", "open", "close", "high", "low", "volume", "turnover"],
  )
  # Normalize numeric columns before timestamp conversion to avoid pandas future warnings.
  df["time"] = pd.to_numeric(df["time"], errors="coerce")
  df["timestamp"] = pd.to_datetime(df["time"], unit="s", errors="coerce")
  for col in ["open", "close", "high", "low", "volume", "turnover"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
  df = df.dropna(subset=["timestamp", "close", "high", "low", "volume"]).sort_values("timestamp").reset_index(drop=True)
  return df


def _compute_true_range(df: pd.DataFrame) -> pd.Series:
  prev_close = df["close"].shift(1)
  range1 = df["high"] - df["low"]
  range2 = (df["high"] - prev_close).abs()
  range3 = (df["low"] - prev_close).abs()
  return pd.concat([range1, range2, range3], axis=1).max(axis=1)


def compute_indicators(df: pd.DataFrame, settings: IndicatorSettings | None = None) -> pd.DataFrame:
  settings = settings or IndicatorSettings()
  out = df.copy()

  # EMA and MACD
  out["ema_fast"] = out["close"].ewm(span=settings.ema_fast, adjust=False).mean()
  out["ema_slow"] = out["close"].ewm(span=settings.ema_slow, adjust=False).mean()
  out["macd"] = out["ema_fast"] - out["ema_slow"]
  out["macd_signal"] = out["macd"].ewm(span=settings.ema_signal, adjust=False).mean()
  out["macd_hist"] = out["macd"] - out["macd_signal"]

  # RSI
  delta = out["close"].diff()
  gain = delta.clip(lower=0)
  loss = (-delta).clip(lower=0)
  avg_gain = gain.ewm(alpha=1 / settings.rsi_length, adjust=False).mean()
  avg_loss = loss.ewm(alpha=1 / settings.rsi_length, adjust=False).mean()
  # Avoid silent downcasting warning by using mask instead of replace -> ffill.
  avg_loss = avg_loss.mask(avg_loss == 0).ffill()
  rs = avg_gain / avg_loss
  out["rsi"] = 100 - (100 / (1 + rs))

  # ATR
  out["true_range"] = _compute_true_range(out)
  out["atr"] = out["true_range"].rolling(window=settings.atr_length, min_periods=settings.atr_length).mean()

  # Bollinger Bands
  out["bb_mid"] = out["close"].rolling(window=settings.bb_length, min_periods=settings.bb_length).mean()
  out["bb_std"] = out["close"].rolling(window=settings.bb_length, min_periods=settings.bb_length).std(ddof=0)
  out["bb_upper"] = out["bb_mid"] + settings.bb_stddev * out["bb_std"]
  out["bb_lower"] = out["bb_mid"] - settings.bb_stddev * out["bb_std"]

  # VWAP
  typical_price = (out["high"] + out["low"] + out["close"]) / 3
  vol_cum = out["volume"].cumsum().replace(0, pd.NA)
  out["vwap"] = (typical_price * out["volume"]).cumsum() / vol_cum

  return out


def _round(val: Any, decimals: int = 4) -> float | None:
  try:
    return round(float(val), decimals)
  except Exception:
    return None


def summarize_interval(df: pd.DataFrame, interval: str) -> Dict[str, Any]:
  enriched = compute_indicators(df)
  latest = enriched.iloc[-1]

  close = float(latest["close"])
  atr = float(latest["atr"]) if pd.notna(latest["atr"]) else None
  atr_pct = (atr / close * 100) if atr and close else None
  bb_mid = latest["bb_mid"] if pd.notna(latest["bb_mid"]) else None
  vwap = latest["vwap"] if pd.notna(latest["vwap"]) else None

  bias = "bullish" if (latest["ema_fast"] > latest["ema_slow"]) and (latest["rsi"] >= 50) else "bearish"
  if latest["ema_fast"] > latest["ema_slow"] and 45 <= latest["rsi"] < 50:
    bias = "neutral-to-bullish"
  elif latest["ema_fast"] < latest["ema_slow"] and 50 > latest["rsi"] >= 45:
    bias = "neutral-to-bearish"

  volatility = "elevated" if atr_pct and atr_pct >= 3 else "normal"
  bb_pos = None
  if bb_mid:
    bb_pos = (close - bb_mid) / bb_mid * 100

  commentary: list[str] = []
  if bias.startswith("bullish"):
    commentary.append("Momentum leaning bullish (fast EMA above slow, RSI>50).")
  elif bias.startswith("bearish"):
    commentary.append("Momentum leaning bearish (fast EMA below slow or RSI<50).")
  else:
    commentary.append("Momentum mixed; wait for confirmation.")
  if atr_pct:
    commentary.append(f"ATR {atr_pct:.2f}% of price ({'high' if volatility=='elevated' else 'moderate'} volatility).")
  if vwap and close > vwap:
    commentary.append("Price above VWAP; bulls in control intraday.")
  elif vwap and close < vwap:
    commentary.append("Price below VWAP; sellers dominate intraday.")
  if latest.get("macd_hist") and abs(latest["macd_hist"]) < 0.1:
    commentary.append("MACD histogram near flat; avoid chasing.")
  if bb_pos is not None and abs(bb_pos) > 2.5:
    commentary.append("Price stretched from Bollinger midline; watch for mean reversion.")

  return {
    "interval": interval,
    "close": _round(close, 6),
    "ema_fast": _round(latest["ema_fast"]),
    "ema_slow": _round(latest["ema_slow"]),
    "rsi": _round(latest["rsi"]),
    "macd": _round(latest["macd"]),
    "macd_signal": _round(latest["macd_signal"]),
    "macd_hist": _round(latest["macd_hist"]),
    "atr": _round(atr),
    "atr_pct": _round(atr_pct, 3) if atr_pct is not None else None,
    "bb_upper": _round(latest["bb_upper"]),
    "bb_lower": _round(latest["bb_lower"]),
    "bb_mid": _round(bb_mid),
    "vwap": _round(vwap),
    "trend_bias": bias,
    "volatility": volatility,
    "commentary": " ".join(commentary),
  }


def summarize_multi_timeframe(snapshots: List[Dict[str, Any]]) -> Dict[str, Any]:
  bullish = sum(1 for s in snapshots if s["trend_bias"].startswith("bullish"))
  bearish = sum(1 for s in snapshots if s["trend_bias"].startswith("bearish"))

  bias = "mixed"
  if bullish == len(snapshots):
    bias = "bullish"
  elif bearish == len(snapshots):
    bias = "bearish"

  vol_flags = [s.get("volatility") for s in snapshots]
  volatility = "elevated" if any(v == "elevated" for v in vol_flags) else "normal"

  entry_hint = "Wait for alignment across timeframes before entering."
  if bias == "bullish":
    entry_hint = "Look for pullbacks above VWAP or mid-Bollinger with RSI>50; avoid chasing if ATR% is high."
  elif bias == "bearish":
    entry_hint = "Consider fades below VWAP with lower highs; size down if volatility is elevated."

  return {"overall_bias": bias, "volatility": volatility, "entry_hint": entry_hint}
