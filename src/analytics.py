from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import pandas as pd

# Kucoin candle fields: [time, open, close, high, low, volume, turnover]
INTERVAL_SECONDS: Dict[str, int] = {
  "1min": 60,
  "5min": 300,
  "15min": 900,
  "30min": 1800,
  "1hour": 3600,
  "2hour": 7200,
  "4hour": 14400,
  "8hour": 28800,
  "12hour": 43200,
  "1day": 86400,
  "1week": 604800,
}


@dataclass
class IndicatorSettings:
  ema_fast: int = 12
  ema_slow: int = 26
  ema_signal: int = 9
  rsi_length: int = 14
  atr_length: int = 14
  bb_length: int = 20
  bb_stddev: float = 2.0
  adx_length: int = 14
  stoch_k_length: int = 14
  stoch_d_length: int = 3


def _as_epoch_seconds(as_of: Any | None) -> float:
  if as_of is None:
    return float(pd.Timestamp.now(tz="UTC").timestamp())
  if isinstance(as_of, (int, float)) and not isinstance(as_of, bool):
    value = float(as_of)
    # Accept exchange-style millisecond timestamps as well as epoch seconds.
    return value / 1000.0 if abs(value) >= 1_000_000_000_000 else value
  timestamp = pd.Timestamp(as_of)
  if timestamp.tzinfo is None:
    timestamp = timestamp.tz_localize("UTC")
  else:
    timestamp = timestamp.tz_convert("UTC")
  return float(timestamp.timestamp())


def _candle_start_seconds(df: pd.DataFrame) -> pd.Series:
  if "time" in df.columns:
    return pd.to_numeric(df["time"], errors="coerce")
  if "timestamp" in df.columns:
    timestamps = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    return timestamps.map(lambda value: value.timestamp() if pd.notna(value) else float("nan"))
  raise ValueError("Candle data must contain a 'time' or 'timestamp' column.")


def exclude_open_candles(
  df: pd.DataFrame,
  interval: str,
  as_of: Any | None = None,
) -> pd.DataFrame:
  """Return only candles whose full interval had elapsed by ``as_of``.

  Exchange candle timestamps identify the start of a bar.  The current bar can
  repaint until ``start + interval`` and must not be used as a completed signal.
  ``as_of`` accepts epoch seconds, epoch milliseconds, or a datetime-like value.
  """
  if interval not in INTERVAL_SECONDS:
    raise ValueError(f"Unsupported candle interval: {interval}")
  starts = _candle_start_seconds(df)
  closed = starts.notna() & (starts + INTERVAL_SECONDS[interval] <= _as_epoch_seconds(as_of))
  return df.loc[closed].copy().reset_index(drop=True)


def validate_candle_data(
  df: pd.DataFrame,
  interval: str,
  as_of: Any | None = None,
  *,
  max_staleness_intervals: float = 1.5,
  max_gap_intervals: float = 1.5,
) -> Dict[str, Any]:
  """Report timestamp freshness, gaps, and OHLC integrity without mutating data."""
  if interval not in INTERVAL_SECONDS:
    raise ValueError(f"Unsupported candle interval: {interval}")
  if max_staleness_intervals < 0 or max_gap_intervals < 1:
    raise ValueError("Candle validation thresholds are out of range.")

  interval_seconds = INTERVAL_SECONDS[interval]
  now_seconds = _as_epoch_seconds(as_of)
  errors: list[str] = []
  invalid_ohlc_rows: list[Any] = []
  gap_count = 0
  max_gap_seconds = 0.0
  latest_close_seconds: float | None = None

  if df.empty:
    errors.append("no candles")
    starts = pd.Series(dtype="float64")
  else:
    starts = _candle_start_seconds(df)
    if starts.isna().any():
      errors.append("invalid candle timestamp")
    valid_starts = starts.dropna().sort_values()
    if valid_starts.duplicated().any():
      errors.append("duplicate candle timestamp")
    if not valid_starts.empty:
      differences = valid_starts.drop_duplicates().diff().dropna()
      if not differences.empty:
        max_gap_seconds = float(differences.max())
        gap_count = int((differences > interval_seconds * max_gap_intervals).sum())
        if gap_count:
          errors.append(f"{gap_count} candle gap(s)")

      closed_starts = valid_starts[valid_starts + interval_seconds <= now_seconds]
      if closed_starts.empty:
        errors.append("no closed candles")
      else:
        latest_close_seconds = float(closed_starts.iloc[-1] + interval_seconds)
        stale_by = max(0.0, now_seconds - latest_close_seconds)
        if stale_by > interval_seconds * max_staleness_intervals:
          errors.append(f"latest closed candle is stale by {stale_by:.0f}s")

  required_ohlc = ("open", "high", "low", "close")
  missing_ohlc = [column for column in required_ohlc if column not in df.columns]
  if missing_ohlc:
    errors.append(f"missing OHLC column(s): {', '.join(missing_ohlc)}")
  elif not df.empty:
    prices = df.loc[:, required_ohlc].apply(pd.to_numeric, errors="coerce")
    finite = prices.apply(lambda column: column.map(lambda value: pd.notna(value) and math.isfinite(float(value))))
    valid = finite.all(axis=1)
    valid &= prices["open"] > 0
    valid &= prices["close"] > 0
    valid &= prices["low"] > 0
    valid &= prices["high"] >= prices[["open", "close", "low"]].max(axis=1)
    valid &= prices["low"] <= prices[["open", "close", "high"]].min(axis=1)
    invalid_ohlc_rows = df.index[~valid].tolist()
    if invalid_ohlc_rows:
      errors.append(f"invalid OHLC in {len(invalid_ohlc_rows)} row(s)")

  return {
    "valid": not errors,
    "errors": errors,
    "rows": len(df),
    "closed_rows": int((starts.notna() & (starts + interval_seconds <= now_seconds)).sum()),
    "open_rows": int((starts.notna() & (starts + interval_seconds > now_seconds)).sum()),
    "gap_count": gap_count,
    "max_gap_seconds": max_gap_seconds,
    "invalid_ohlc_rows": invalid_ohlc_rows,
    "latest_close_time": latest_close_seconds,
  }


def candles_to_dataframe(
  candles: Sequence[Sequence[Any]],
  interval: str | None = None,
  *,
  as_of: Any | None = None,
  closed_only: bool = False,
) -> pd.DataFrame:
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
  if closed_only:
    if interval is None:
      raise ValueError("interval is required when closed_only=True")
    df = exclude_open_candles(df, interval, as_of=as_of)
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
  rs = avg_gain / avg_loss
  rsi = 100 - (100 / (1 + rs))
  no_gain = avg_gain.eq(0)
  no_loss = avg_loss.eq(0)
  # Define the zero-denominator cases explicitly: uninterrupted gains/losses sit at
  # the corresponding extreme, while an unchanged market has neutral momentum.
  rsi = rsi.mask(avg_gain.gt(0) & no_loss, 100.0)
  rsi = rsi.mask(no_gain & avg_loss.gt(0), 0.0)
  out["rsi"] = rsi.mask(no_gain & no_loss, 50.0)

  # ATR
  out["true_range"] = _compute_true_range(out)
  out["atr"] = out["true_range"].rolling(window=settings.atr_length, min_periods=settings.atr_length).mean()

  # ADX (Average Directional Index)
  prev_high = out["high"].shift(1)
  prev_low = out["low"].shift(1)
  plus_dm = (out["high"] - prev_high).clip(lower=0)
  minus_dm = (prev_low - out["low"]).clip(lower=0)
  plus_dm = plus_dm.where(plus_dm > minus_dm, 0.0)
  minus_dm = minus_dm.where(minus_dm > plus_dm, 0.0)
  wilder_alpha = 1.0 / settings.adx_length
  smooth_plus_dm = plus_dm.ewm(alpha=wilder_alpha, adjust=False).mean()
  smooth_minus_dm = minus_dm.ewm(alpha=wilder_alpha, adjust=False).mean()
  smooth_tr = out["true_range"].ewm(alpha=wilder_alpha, adjust=False).mean()
  smooth_tr_safe = smooth_tr.replace(0, float("nan"))
  plus_di = 100 * smooth_plus_dm / smooth_tr_safe
  minus_di = 100 * smooth_minus_dm / smooth_tr_safe
  out["plus_di"] = plus_di
  out["minus_di"] = minus_di
  di_sum = (plus_di + minus_di).replace(0, float("nan"))
  di_diff = (plus_di - minus_di).abs()
  dx = 100 * di_diff / di_sum
  out["adx"] = dx.ewm(alpha=wilder_alpha, adjust=False).mean()

  # Bollinger Bands
  out["bb_mid"] = out["close"].rolling(window=settings.bb_length, min_periods=settings.bb_length).mean()
  out["bb_std"] = out["close"].rolling(window=settings.bb_length, min_periods=settings.bb_length).std(ddof=0)
  out["bb_upper"] = out["bb_mid"] + settings.bb_stddev * out["bb_std"]
  out["bb_lower"] = out["bb_mid"] - settings.bb_stddev * out["bb_std"]

  # Bollinger Band Width (percentage)
  out["bbw"] = ((out["bb_upper"] - out["bb_lower"]) / out["bb_mid"].replace(0, float("nan"))) * 100

  # Stochastic Oscillator
  stoch_low = out["low"].rolling(window=settings.stoch_k_length, min_periods=settings.stoch_k_length).min()
  stoch_high = out["high"].rolling(window=settings.stoch_k_length, min_periods=settings.stoch_k_length).max()
  stoch_range = (stoch_high - stoch_low).replace(0, float("nan"))
  out["stoch_k"] = 100 * (out["close"] - stoch_low) / stoch_range
  out["stoch_d"] = out["stoch_k"].rolling(window=settings.stoch_d_length, min_periods=settings.stoch_d_length).mean()

  # VWAP
  typical_price = (out["high"] + out["low"] + out["close"]) / 3
  vol_cum = out["volume"].cumsum().replace(0, pd.NA)
  out["vwap"] = (typical_price * out["volume"]).cumsum() / vol_cum

  return out


def compute_volume_profile(df: pd.DataFrame, num_bins: int = 50) -> Dict[str, Any]:
  high_max = df["high"].max()
  low_min = df["low"].min()
  price_range = high_max - low_min
  if price_range <= 0 or len(df) == 0:
    return {"poc": None, "vah": None, "val": None}
  bin_size = price_range / num_bins
  bins: Dict[float, float] = {}
  for _, row in df.iterrows():
    row_range = row["high"] - row["low"]
    if row_range <= 0:
      continue
    vol_per_unit = row["volume"] / row_range
    lo = row["low"]
    hi = row["high"]
    b = math.floor(lo / bin_size) * bin_size
    while b < hi:
      overlap = min(b + bin_size, hi) - max(b, lo)
      if overlap > 0:
        bins[b] = bins.get(b, 0.0) + vol_per_unit * overlap
      b += bin_size
  if not bins:
    return {"poc": None, "vah": None, "val": None}
  poc_bin = max(bins, key=bins.get)
  poc = poc_bin + bin_size / 2
  sorted_bins = sorted(bins.items(), key=lambda x: x[1], reverse=True)
  total_vol = sum(v for _, v in sorted_bins)
  cum = 0.0
  va_prices: list[float] = []
  for price, vol in sorted_bins:
    cum += vol
    va_prices.append(price + bin_size / 2)
    if cum >= total_vol * 0.70:
      break
  vah = max(va_prices)
  val = min(va_prices)
  return {"poc": round(poc, 6), "vah": round(vah, 6), "val": round(val, 6)}


def _round(val: Any, decimals: int = 4) -> float | None:
  try:
    return round(float(val), decimals)
  except Exception:
    return None


def classify_regime(
  adx: float | None,
  bbw: float | None,
  atr_pct: float | None,
) -> Dict[str, Any]:
  if adx is None or bbw is None:
    return {"regime": "unknown", "confidence": 0.0, "details": "Insufficient data for regime detection."}

  if bbw < 2.0 and adx < 20:
    return {
      "regime": "squeeze",
      "confidence": min(0.9, 0.5 + (20 - adx) / 40 + (2.0 - bbw) / 4),
      "details": f"Bollinger squeeze (BBW={bbw:.1f}%, ADX={adx:.1f}). Breakout imminent. Avoid new range trades.",
    }

  if adx > 35:
    return {
      "regime": "trending",
      "confidence": min(0.95, 0.6 + (adx - 25) / 50),
      "details": f"Strong trend (ADX={adx:.1f}, BBW={bbw:.1f}%). Use trend-following entries.",
    }

  if adx > 25:
    return {
      "regime": "trending",
      "confidence": min(0.8, 0.5 + (adx - 20) / 30),
      "details": f"Moderate trend (ADX={adx:.1f}, BBW={bbw:.1f}%). Trend-following preferred.",
    }

  if adx < 20 and 2.0 <= bbw <= 6.0:
    return {
      "regime": "ranging",
      "confidence": min(0.85, 0.5 + (20 - adx) / 30 + 0.1),
      "details": f"Range-bound market (ADX={adx:.1f}, BBW={bbw:.1f}%). Mean-reversion entries preferred.",
    }

  if adx < 20:
    return {
      "regime": "ranging",
      "confidence": 0.55,
      "details": f"Low directional movement (ADX={adx:.1f}, BBW={bbw:.1f}%). Lean toward range strategies.",
    }

  if adx < 22:
    return {
      "regime": "ranging",
      "confidence": 0.5,
      "details": f"Weak directional signal (ADX={adx:.1f}, BBW={bbw:.1f}%). Lean toward range strategies with caution.",
    }

  return {
    "regime": "trending",
    "confidence": 0.5,
    "details": f"Ambiguous regime (ADX={adx:.1f}, BBW={bbw:.1f}%). Use reduced size either way.",
  }


def summarize_interval(df: pd.DataFrame, interval: str) -> Dict[str, Any]:
  enriched = compute_indicators(df)
  latest = enriched.iloc[-1]
  prev = enriched.iloc[-2] if len(enriched) >= 2 else None

  close = float(latest["close"])
  atr = float(latest["atr"]) if pd.notna(latest["atr"]) else None
  atr_pct = (atr / close * 100) if atr and close else None
  bb_mid = latest["bb_mid"] if pd.notna(latest["bb_mid"]) else None
  vwap = latest["vwap"] if pd.notna(latest["vwap"]) else None
  adx = float(latest["adx"]) if pd.notna(latest.get("adx")) else None
  bbw = float(latest["bbw"]) if pd.notna(latest.get("bbw")) else None
  bbw_prev = float(prev["bbw"]) if prev is not None and pd.notna(prev.get("bbw")) else None
  plus_di = float(latest["plus_di"]) if pd.notna(latest.get("plus_di")) else None
  minus_di = float(latest["minus_di"]) if pd.notna(latest.get("minus_di")) else None
  stoch_k = float(latest["stoch_k"]) if pd.notna(latest.get("stoch_k")) else None
  stoch_d = float(latest["stoch_d"]) if pd.notna(latest.get("stoch_d")) else None
  regime = classify_regime(adx, bbw, atr_pct)

  ema_bullish = latest["ema_fast"] > latest["ema_slow"]
  rsi = latest["rsi"]
  if ema_bullish and rsi >= 45:
    bias = "bullish"
  elif not ema_bullish and rsi <= 55:
    bias = "bearish"
  elif ema_bullish and 40 <= rsi < 45:
    bias = "neutral-to-bullish"
  elif not ema_bullish and 55 < rsi <= 60:
    bias = "neutral-to-bearish"
  else:
    bias = "neutral"

  volatility = "elevated" if atr_pct and atr_pct >= 3 else "normal"
  bb_pos = None
  if bb_mid:
    bb_pos = (close - bb_mid) / bb_mid * 100

  commentary: list[str] = []
  if bias.startswith("bullish"):
    commentary.append("Momentum leaning bullish (fast EMA above slow, RSI supportive).")
  elif bias.startswith("bearish"):
    commentary.append("Momentum leaning bearish (fast EMA below slow, RSI confirming).")
  elif bias.startswith("neutral-to"):
    commentary.append("Momentum transitioning; early directional signal, confirm with price action.")
  else:
    commentary.append("Momentum neutral; no clear directional edge.")
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

  if regime["regime"] == "ranging":
    commentary.append(f"RANGE DETECTED (ADX={adx:.1f}). Mean-reversion entries at BB bands preferred.")
    if stoch_k is not None:
      if stoch_k < 20:
        commentary.append(f"Stochastic oversold ({stoch_k:.0f}); range-buy signal near BB lower band.")
      elif stoch_k > 80:
        commentary.append(f"Stochastic overbought ({stoch_k:.0f}); range-sell/short signal near BB upper band.")
  elif regime["regime"] == "squeeze":
    commentary.append(f"BB SQUEEZE (BBW={bbw:.1f}%). Breakout imminent; avoid new positions, prepare to catch breakout direction.")
  elif regime["regime"] == "trending" and adx and adx > 35:
    commentary.append(f"STRONG TREND (ADX={adx:.1f}). Avoid mean-reversion trades; ride the trend with trailing stops.")

  if len(enriched) >= 3:
    recent_adx = enriched["adx"].dropna().tail(3)
    if len(recent_adx) >= 3:
      adx_delta = float(recent_adx.iloc[-1]) - float(recent_adx.iloc[0])
      if regime["regime"] == "ranging" and adx_delta > 3:
        commentary.append(f"WARNING: ADX rising (+{adx_delta:.1f}). Range may be ending; prepare for breakout.")
      elif regime["regime"] == "trending" and adx_delta < -3:
        commentary.append(f"ADX weakening ({adx_delta:.1f}). Trend may be exhausting; watch for range formation.")

  vol_profile = compute_volume_profile(df)

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
    "bbw": _round(bbw, 3),
    "bbw_prev": _round(bbw_prev, 3) if bbw_prev is not None else None,
    "vwap": _round(vwap),
    "adx": _round(adx),
    "plus_di": _round(plus_di),
    "minus_di": _round(minus_di),
    "stoch_k": _round(stoch_k),
    "stoch_d": _round(stoch_d),
    "market_regime": regime,
    "trend_bias": bias,
    "volatility": volatility,
    "volume_profile": vol_profile,
    "commentary": " ".join(commentary),
  }


_TF_WEIGHTS = {"4hour": 0.40, "1hour": 0.35, "15min": 0.25}
_DAILY_INTERVAL = "1day"


def summarize_multi_timeframe(snapshots: List[Dict[str, Any]]) -> Dict[str, Any]:
  ordered = sorted(snapshots, key=lambda s: INTERVAL_SECONDS.get(s.get("interval", ""), 0), reverse=True)

  # Separate daily snapshot (regime gate) from intraday snapshots (weighted scoring)
  daily_snap = None
  intraday = []
  for snap in ordered:
    if snap.get("interval") == _DAILY_INTERVAL:
      daily_snap = snap
    else:
      intraday.append(snap)

  primary = intraday[0] if intraday else {}
  secondary = intraday[1] if len(intraday) > 1 else None
  tertiary = intraday[2] if len(intraday) > 2 else None

  def _direction(bias: str) -> str:
    if bias.startswith("bullish") or bias == "neutral-to-bullish":
      return "bullish"
    if bias.startswith("bearish") or bias == "neutral-to-bearish":
      return "bearish"
    return "neutral"

  def _dir_score(direction: str) -> float:
    if direction == "bullish":
      return 1.0
    if direction == "bearish":
      return -1.0
    return 0.0

  all_dirs = []
  for snap in intraday:
    interval = snap.get("interval", "")
    direction = _direction(snap.get("trend_bias", "neutral"))
    weight = _TF_WEIGHTS.get(interval, 0.2)
    all_dirs.append((interval, direction, weight))

  weighted_score = sum(_dir_score(d) * w for _, d, w in all_dirs)
  total_weight = sum(w for _, _, w in all_dirs) or 1.0
  normalized_score = weighted_score / total_weight

  if normalized_score > 0.3:
    overall_bias = "bullish"
  elif normalized_score < -0.3:
    overall_bias = "bearish"
  else:
    overall_bias = "neutral"

  primary_dir = _direction(primary.get("trend_bias", "neutral"))
  secondary_dir = _direction(secondary.get("trend_bias", "neutral")) if secondary else "neutral"
  tertiary_dir = _direction(tertiary.get("trend_bias", "neutral")) if tertiary else "neutral"

  agreeing = sum(1 for _, d, _ in all_dirs if d == overall_bias and d != "neutral")
  total_non_neutral = sum(1 for _, d, _ in all_dirs if d != "neutral")
  conflicting = sum(1 for _, d, _ in all_dirs if d != "neutral" and d != overall_bias)

  if agreeing >= 2 and conflicting == 0:
    strength = "strong"
  elif agreeing >= 1 and conflicting == 0:
    strength = "moderate"
  elif conflicting > 0 and agreeing > conflicting:
    strength = "moderate"
  elif overall_bias != "neutral":
    strength = "weak"
  else:
    strength = "weak"
    overall_bias = "neutral"

  # Daily gate: if 1D bias opposes the intraday bias, downgrade; if it agrees, boost
  daily_bias = "neutral"
  daily_bias_raw = "neutral"
  daily_gate_applied = False
  daily_exhausted = False
  daily_trend_weak = False
  if daily_snap:
    daily_bias = _direction(daily_snap.get("trend_bias", "neutral"))
    # RSI exhaustion and weak/choppy ADX have different trading implications.
    # Both make the daily direction unsuitable as a hard gate, but low ADX must
    # not trigger anti-FOMO rules intended for an extended directional move.
    daily_rsi = daily_snap.get("rsi")
    daily_adx = daily_snap.get("adx")
    if daily_bias == "bullish" and daily_rsi is not None and daily_rsi >= 70:
      daily_exhausted = True
    elif daily_bias == "bearish" and daily_rsi is not None and daily_rsi <= 30:
      daily_exhausted = True
    if daily_adx is not None and daily_adx < 18:
      daily_trend_weak = True
    daily_bias_raw = daily_bias
    if daily_exhausted or daily_trend_weak:
      daily_bias = "neutral"
    if daily_bias != "neutral" and overall_bias != "neutral" and daily_bias != overall_bias:
      overall_bias = "neutral"
      strength = "weak"
      daily_gate_applied = True
    elif daily_bias != "neutral" and daily_bias == overall_bias:
      if strength == "moderate":
        strength = "strong"
      elif strength == "weak":
        strength = "moderate"

  tf_conflict = False
  if len(all_dirs) >= 2:
    higher_dir = all_dirs[0][1]
    lower_dirs = [d for _, d, _ in all_dirs[1:] if d != "neutral"]
    if higher_dir != "neutral" and lower_dirs and any(d != higher_dir for d in lower_dirs):
      tf_conflict = True
  if daily_bias != "neutral" and all_dirs:
    intraday_dirs = [d for _, d, _ in all_dirs if d != "neutral"]
    if intraday_dirs and any(d != daily_bias for d in intraday_dirs):
      tf_conflict = True

  vol_flags = [s.get("volatility") for s in snapshots]
  volatility = "elevated" if any(v == "elevated" for v in vol_flags) else "normal"

  primary_regime = (primary.get("market_regime") or {}).get("regime", "unknown")
  secondary_regime = ((secondary.get("market_regime") or {}).get("regime", "unknown")) if secondary else "unknown"
  primary_regime_conf = (primary.get("market_regime") or {}).get("confidence", 0)
  secondary_regime_conf = ((secondary.get("market_regime") or {}).get("confidence", 0)) if secondary else 0

  if primary_regime == "ranging" and secondary_regime == "ranging":
    overall_regime = "ranging"
    regime_confidence = (primary_regime_conf + secondary_regime_conf) / 2
  elif primary_regime == "squeeze" or secondary_regime == "squeeze":
    overall_regime = "squeeze"
    regime_confidence = max(primary_regime_conf, secondary_regime_conf)
  elif primary_regime == "trending" and secondary_regime in ("trending", "unknown"):
    overall_regime = "trending"
    regime_confidence = primary_regime_conf
  elif primary_regime == "ranging" and secondary_regime == "trending":
    overall_regime = "ranging"
    regime_confidence = primary_regime_conf * 0.7
  elif primary_regime == "trending" and secondary_regime == "ranging":
    overall_regime = "trending"
    regime_confidence = primary_regime_conf * 0.7
  else:
    overall_regime = primary_regime
    regime_confidence = primary_regime_conf * 0.5

  if overall_regime == "ranging":
    entry_hint = (
      "RANGE REGIME: Multiple timeframes confirm range-bound conditions. "
      "Use mean-reversion strategy: "
      "LONG near BB lower band or Volume Profile VAL when Stochastic <20 or RSI <35; "
      "SHORT near BB upper band or Volume Profile VAH when Stochastic >80 or RSI >65. "
      "Target: BB midline or POC. Stop: 1 ATR beyond the BB band you entered at. "
      "Size: 50-70% of normal."
    )
  elif overall_regime == "squeeze":
    entry_hint = (
      "SQUEEZE REGIME: Bollinger Band compression detected. "
      "Avoid range trades; a breakout is imminent. "
      "Prepare to enter in the breakout direction once price closes beyond BB band with rising ADX."
    )
  elif overall_bias == "bullish":
    if strength == "strong":
      entry_hint = "Strong bullish alignment across timeframes; look for pullbacks to VWAP, POC, or mid-Bollinger for long entries."
    elif strength == "moderate":
      entry_hint = "Moderate bullish bias; higher TF bullish but lower TF not fully confirmed. Consider reduced size long on pullback."
    else:
      entry_hint = "Weak bullish signal with timeframe conflict; use smaller size, tight stops, favor scalps over swings."
  elif overall_bias == "bearish":
    if strength == "strong":
      entry_hint = "Strong bearish alignment across timeframes; look for rallies to VWAP, POC, or mid-Bollinger for short entries."
    elif strength == "moderate":
      entry_hint = "Moderate bearish bias; higher TF bearish but lower TF not fully confirmed. Consider reduced size short near resistance."
    else:
      entry_hint = "Weak bearish signal with timeframe conflict; use smaller size, tight stops, favor scalps over swings."
  else:
    entry_hint = "No clear directional bias; consider range-bound strategies or reduce size significantly."

  if daily_exhausted:
    if daily_bias_raw == "bearish":
      entry_hint += (
        " DAILY BEARISH EXHAUSTED: do not chase a fresh dump, but a high-confidence continuation "
        "SHORT is eligible after a rally/retest when both 1h and 15m are bearish; the code gate "
        "will enforce confirmation and reduced sizing."
      )
    elif daily_bias_raw == "bullish":
      entry_hint += " DAILY EXHAUSTED: 1D bullish RSI extreme — do NOT open continuation longs. Counter-trend only with a strong reversal signal."
    else:
      entry_hint += " DAILY EXHAUSTED: 1D RSI is extreme — avoid chasing the extended move."
  elif daily_trend_weak:
    entry_hint += " DAILY TREND WEAK: 1D ADX is below 18, so its directional gate is neutralized; favor range-aware confirmation."
  elif daily_gate_applied:
    entry_hint += f" DAILY GATE: 1D trend is {daily_bias} — opposing intraday bias was overridden to neutral. Do NOT open counter-daily trades."
  elif daily_bias != "neutral":
    entry_hint += f" Daily trend confirms: 1D bias is {daily_bias}."

  if tf_conflict:
    entry_hint += " WARNING: Timeframe conflict detected — higher and lower timeframes disagree. Reduce size or wait for alignment."

  if overall_regime == "trending":
    entry_hint += " [Trend regime confirmed by ADX.]"

  if volatility == "elevated":
    entry_hint += " Volatility elevated — widen stops and reduce size."

  squeeze_breakout: str | None = None
  h1 = next((s for s in snapshots if s.get("interval") == "1hour"), None)
  if h1:
    bbw_now = h1.get("bbw")
    bbw_prev_h1 = h1.get("bbw_prev")
    adx_now = h1.get("adx")
    rsi_now = h1.get("rsi")
    close_now = h1.get("close")
    bb_upper = h1.get("bb_upper")
    bb_lower = h1.get("bb_lower")
    if (
      bbw_prev_h1 is not None and bbw_now is not None
      and bbw_prev_h1 < 2.0 and bbw_now > bbw_prev_h1 * 1.25
      and adx_now is not None and adx_now > 20
      and close_now is not None and rsi_now is not None
    ):
      if bb_upper is not None and close_now > bb_upper and rsi_now > 55:
        squeeze_breakout = "long"
      elif bb_lower is not None and close_now < bb_lower and rsi_now < 45:
        squeeze_breakout = "short"
  if squeeze_breakout:
    entry_hint += (
      f" SQUEEZE BREAKOUT ({squeeze_breakout.upper()}): 1h BBW expanded off the floor with ADX>20 and price beyond BB band. "
      "Confirm with volume ≥ 1.5× 20-candle average before entering."
    )

  intraday_bias_15m = _direction(
    next((s.get("trend_bias", "neutral") for s in snapshots if s.get("interval") == "15min"), "neutral")
  )
  intraday_bias_1h = _direction(
    next((s.get("trend_bias", "neutral") for s in snapshots if s.get("interval") == "1hour"), "neutral")
  )
  intraday_bias_4h = _direction(
    next((s.get("trend_bias", "neutral") for s in snapshots if s.get("interval") == "4hour"), "neutral")
  )

  return {
    "overall_bias": overall_bias,
    "strength": strength,
    "weighted_score": _round(normalized_score, 3),
    "daily_bias": daily_bias,
    "daily_bias_raw": daily_bias_raw,
    "daily_gate_applied": daily_gate_applied,
    "daily_exhausted": daily_exhausted,
    "daily_trend_weak": daily_trend_weak,
    "timeframe_conflict": tf_conflict,
    "intraday_bias_15m": intraday_bias_15m,
    "intraday_bias_1h": intraday_bias_1h,
    "intraday_bias_4h": intraday_bias_4h,
    "volatility": volatility,
    "market_regime": overall_regime,
    "regime_confidence": _round(regime_confidence, 3),
    "primary_regime": primary.get("market_regime"),
    "secondary_regime": secondary.get("market_regime") if secondary else None,
    "squeeze_breakout": squeeze_breakout,
    "entry_hint": entry_hint,
    "primary_interval": primary.get("interval"),
    "secondary_interval": secondary.get("interval") if secondary else None,
    "tertiary_interval": tertiary.get("interval") if tertiary else None,
    "daily_interval": _DAILY_INTERVAL if daily_snap else None,
    "volume_profile": primary.get("volume_profile"),
  }
