"""Agent tool definitions for trAIde — extracted from agent.py.

Every tool is a closure over per-run state: config, exchange clients, the MemoryStore, the live
TradingSnapshot, and a set of helper closures built in ``run_trading_agent``. ``build_tools(ctx)``
rebinds that context to locals so the tool bodies are byte-identical to their original form, then
returns them as a namespace. Sections below, in order: spot trading, market data & analysis, futures
trading, futures/account data & screening, transfers & account, planning/memory/curation, and news.

Import note: this module imports a few low-level helpers from ``agent`` and ``agent`` imports
``build_tools`` lazily (inside ``run_trading_agent``), so there is no import cycle at load time.
"""

from __future__ import annotations

import math
import time
import uuid
from types import SimpleNamespace
from typing import Any, Dict, List

import requests

from agents.tool import function_tool
from .analytics import (
  INTERVAL_SECONDS,
  candles_to_dataframe,
  compute_indicators,
  summarize_interval,
  summarize_multi_timeframe,
)
from .kucoin import KucoinFuturesOrderRequest, KucoinOrderRequest
from .protection import should_block_chase
from .regime import (
  allow_reversal_long,
  allow_trend_aligned_short,
  block_alt_long_in_btc_downtrend,
  concentration_scale,
  conviction_size_factor,
  effective_min_confidence,
  regime_size_factor,
  resolve_gate_deadlock,
  reward_risk_ratio,
)
from .utils import normalize_symbol as _normalize_symbol
from .agent import (
  logger,
  _aggregate_account_totals,
  _base_currency,
  _build_compact_account_state,
  _resolve_allowed_spot_symbol,
  _screen_contracts,
  _to_float,
  _to_futures_symbol,
  _truncate_to_increment,
)


def build_tools(ctx: SimpleNamespace) -> SimpleNamespace:
  """Instantiate all agent tools bound to the given per-run context. Returns a namespace of tools."""
  cfg = ctx.cfg
  kucoin = ctx.kucoin
  kucoin_futures = ctx.kucoin_futures
  memory = ctx.memory
  snapshot = ctx.snapshot
  allowed_symbols = ctx.allowed_symbols
  balances_by_currency = ctx.balances_by_currency
  fees = ctx.fees
  _daily_gate_state = ctx._daily_gate_state
  _futures_margin_mode = ctx._futures_margin_mode
  _apply_cross_leverage = ctx._apply_cross_leverage
  _btc_daily_bias = ctx._btc_daily_bias
  _edge_state = ctx._edge_state
  _fee_adjusted_breakeven = ctx._fee_adjusted_breakeven
  _get_contract_spec = ctx._get_contract_spec
  _repair_allowed_symbol = ctx._repair_allowed_symbol
  _spot_position_info = ctx._spot_position_info
  _spot_position_size = ctx._spot_position_size
  _stop_distance_ok = ctx._stop_distance_ok


  # ── Spot trading — market/limit orders, stops & position protection ─────────────────────────────
  @function_tool
  async def place_market_order(
    symbol: str,
    side: str,
    funds: float,
    confidence: float | None = None,
    rationale: str | None = None,
    stop_price: float | None = None,
    take_profit_price: float | None = None,
    risk_pct: float | None = None,
    atr_multiple: float | None = None,
    target_rr: float | None = None,
    auto_protect: bool = True,
  ) -> Dict[str, Any]:
    """Place a spot market order on Kucoin using quote funds in USDT. Enforces stop/TP for buys to avoid ultra-tight churn."""
    symbol = _normalize_symbol(symbol)
    if symbol not in allowed_symbols:
      return {"error": "Unsupported symbol", "allowed": sorted(allowed_symbols)}

    # Circuit breaker: block new entries when trading is restricted
    if (side or "").lower() == "buy" and snapshot.trading_restricted:
      return {"rejected": True, "reason": f"Trading restricted (close-only mode): {snapshot.restriction_reason}", "hint": "Only sell/close operations are allowed during circuit breaker activation."}

    # Post-loss cooldown: prevent revenge trading on same symbol
    if (side or "").lower() == "buy" and cfg.trading.post_loss_cooldown_minutes > 0:
      last_loss_ts = memory.last_loss_time(symbol)
      if last_loss_ts:
        elapsed_min = (int(time.time()) - last_loss_ts) / 60
        if elapsed_min < cfg.trading.post_loss_cooldown_minutes:
          remaining = int(cfg.trading.post_loss_cooldown_minutes - elapsed_min)
          return {"rejected": True, "reason": f"Post-loss cooldown active for {symbol} ({remaining}min remaining)", "hint": "Prevents revenge trading after a stop-loss. Try a different symbol or wait for cooldown to expire."}

    # No-chase after a win: don't re-buy a symbol at a worse price right after taking profit.
    if (side or "").lower() == "buy" and cfg.profit_protection.no_chase_enabled:
      win = memory.recent_win_close(symbol, cfg.profit_protection.post_win_cooldown_minutes)
      ref_ticker = snapshot.tickers.get(symbol)
      ref_px = float(ref_ticker.price) if ref_ticker else 0.0
      if win and win.get("exitPrice") and ref_px > 0 and should_block_chase(
        close_type=win.get("closeType") or "",
        exit_price=float(win["exitPrice"]),
        new_side=side,
        new_price=ref_px,
        buffer_pct=cfg.profit_protection.no_chase_buffer_pct,
      ):
        return {"rejected": True, "reason": f"No-chase: just took profit on {symbol} near {float(win['exitPrice']):.6g}; not re-buying at a worse price ({ref_px:.6g})", "hint": "You recently closed this in profit. Re-enter only on a genuine pullback (better than your exit) or after the post-win cooldown expires."}

    # Post-trade cooldown: minimum interval between trades on same symbol
    if (side or "").lower() == "buy" and cfg.trading.min_trade_interval_minutes > 0:
      last_trade_ts = memory.last_trade_time(symbol)
      if last_trade_ts:
        elapsed_min = (int(time.time()) - last_trade_ts) / 60
        if elapsed_min < cfg.trading.min_trade_interval_minutes:
          remaining = int(cfg.trading.min_trade_interval_minutes - elapsed_min)
          return {"rejected": True, "reason": f"Trade interval cooldown for {symbol} ({remaining}min remaining)", "hint": f"Minimum {cfg.trading.min_trade_interval_minutes:.0f}min between trades on same symbol to prevent overtrading."}

    # Daily gate enforcement: block spot buys opposing the 1D trend (unless exhausted)
    if (side or "").lower() == "buy" and symbol in _daily_gate_state:
      gate = _daily_gate_state[symbol]
      daily_bias = gate.get("daily_bias", "neutral")
      daily_bias_raw = gate.get("daily_bias_raw", daily_bias)
      daily_exhausted = gate.get("daily_exhausted", False)
      if daily_exhausted and daily_bias_raw == "bullish":
        logger.warning("ANTI-FOMO BLOCK: spot buy %s rejected — daily bullish exhausted (RSI extreme)", symbol)
        return {"rejected": True, "reason": "Daily exhaustion: bullish trend overextended — no continuation entry", "hint": "Daily RSI is at an extreme high. Wait for the pullback or trade counter-trend with a clear reversal signal."}
      if daily_bias == "bearish" and not daily_exhausted:
        logger.warning("DAILY GATE BLOCK: spot buy %s rejected — 1D trend is bearish", symbol)
        return {"rejected": True, "reason": f"Daily gate: 1D trend is bearish — spot buy blocked", "daily_bias": daily_bias, "hint": "The 1D timeframe is bearish. Spot buys are blocked until the daily trend turns neutral or bullish."}
      # 1h alignment: block longs when 1h is bearish (catches buying bounces in confirmed corrections)
      intraday_bias_1h = gate.get("intraday_bias_1h", "neutral")
      if intraday_bias_1h == "bearish":
        logger.warning("1H ALIGN BLOCK: spot buy %s rejected — 1h bias is bearish", symbol)
        return {"rejected": True, "reason": "1h trend is bearish — long entry blocked", "hint": "1h timeframe is in a downtrend. The daily uptrend is in correction, not a healthy pullback. Wait for 1h to turn neutral/bullish or pick a different symbol."}
      # Timeframe-conflict gate: catches 15m vs higher-TF disagreement not already blocked by 1h alignment
      tf_conflict = gate.get("timeframe_conflict", False)
      intraday_bias_15m = gate.get("intraday_bias_15m", "neutral")
      if tf_conflict and intraday_bias_15m == "bearish":
        logger.warning("TF CONFLICT BLOCK: spot buy %s rejected — daily/intraday split, 15m bearish opposes buy", symbol)
        return {"rejected": True, "reason": "Timeframe conflict: 15m bearish opposes proposed buy", "hint": "Wait for 15m to align with the higher-TF bias, or pick a different symbol."}

    # Volatility filter: soft-scale position below 1.5× threshold; hard-block above
    _atr_scale_m = 1.0
    if (side or "").lower() == "buy" and symbol in _daily_gate_state:
      gate = _daily_gate_state[symbol]
      daily_atr_pct = gate.get("daily_atr_pct")
      if daily_atr_pct is not None and daily_atr_pct > cfg.trading.max_atr_pct_for_entry:
        hard_limit = cfg.trading.max_atr_pct_for_entry * 1.5
        if daily_atr_pct > hard_limit:
          logger.warning("VOLATILITY BLOCK: spot buy %s rejected — ATR=%.2f%% exceeds hard limit %.2f%%", symbol, daily_atr_pct, hard_limit)
          return {"rejected": True, "reason": f"Extreme volatility: ATR={daily_atr_pct:.2f}% > {hard_limit:.1f}% hard limit", "hint": "Wait for volatility to settle before entering."}
        _atr_scale_m = max(0.30, (cfg.trading.max_atr_pct_for_entry / daily_atr_pct) ** 2)
        logger.info("VOLATILITY SOFT GATE: spot buy %s ATR=%.2f%% — scaling position to %.0f%%", symbol, daily_atr_pct, _atr_scale_m * 100)

    # Confidence enforcement: reject buys below minimum confidence
    if (side or "").lower() == "buy" and confidence is not None and confidence < cfg.trading.min_confidence:
      return {"rejected": True, "reason": f"Confidence {confidence:.2f} below minimum {cfg.trading.min_confidence}", "hint": "Only enter trades with sufficient conviction. Analyze another coin or wait for a better setup."}

    # Correlation gate: block alt spot BUYS while BTC's daily regime is bearish (high-beta blowup guard).
    if block_alt_long_in_btc_downtrend(symbol=symbol, side=side, btc_daily_bias=_btc_daily_bias(), cfg=cfg.regime):
      logger.warning("CORRELATION GATE BLOCK: spot buy %s rejected — BTC daily regime is bearish", symbol)
      return {"rejected": True, "reason": f"Correlation gate: BTC daily regime is bearish — spot buy on alt {symbol} blocked", "hint": "Alts are high-beta to BTC; do NOT buy an altcoin while BTC's daily trend is down. Wait for BTC's daily to turn or trade a major."}

    try:
      funds_val = float(funds or 0)
    except (TypeError, ValueError):
      funds_val = 0.0
    funds_val = funds_val * _atr_scale_m
    if funds_val <= 0 or not symbol:
      return {"error": "Invalid funds or symbol"}
    fee_rate = fees.get("spot_taker", 0.001)
    slippage_rate = cfg.trading.estimated_slippage_pct
    funds_with_fee = funds_val * (1 + fee_rate + slippage_rate)
    if funds_val > snapshot.max_position_usd:
      return {"rejected": True, "reason": "Exceeds maxPositionUsd"}
    trades_today = memory.trades_today(symbol)
    if trades_today >= cfg.trading.max_trades_per_symbol_per_day:
      return {
        "rejected": True,
        "reason": "Daily trade cap reached",
        "tradesToday": trades_today,
        "limit": cfg.trading.max_trades_per_symbol_per_day,
      }
    if cfg.trading.sentiment_filter_enabled:
      latest_sent = memory.latest_sentiment(symbol)
      day_key = int(time.time() // 86400)
      if not latest_sent or latest_sent.get("day") != day_key:
        return {"rejected": True, "reason": "Sentiment missing for today", "minScore": cfg.trading.sentiment_min_score}
      if latest_sent.get("score", 0) < cfg.trading.sentiment_min_score:
        return {
          "rejected": True,
          "reason": "Sentiment below threshold",
          "score": latest_sent.get("score"),
          "minScore": cfg.trading.sentiment_min_score,
        }
    # Refresh balances to reduce stale balance failures.
    fresh_balances = kucoin.get_trade_accounts()
    fresh_by_currency: Dict[str, float] = {}
    for bal in fresh_balances:
      fresh_by_currency[bal.currency] = fresh_by_currency.get(bal.currency, 0.0) + float(bal.available or 0)

    spot_usdt = fresh_by_currency.get("USDT", 0.0)
    futures_available = 0.0
    if cfg.kucoin_futures.enabled and kucoin_futures:
      try:
        overview = kucoin_futures.get_account_overview()
        futures_available = float(overview.get("availableBalance") or 0.0)
      except Exception as exc:
        logger.warning("Futures overview unavailable: %s", exc)

    try:
      mark_price = float(snapshot.tickers[symbol].price)
    except Exception:
      mark_price = 0.0

    transfer_used: dict[str, Any] | None = None
    if side != "sell":
      total_available = spot_usdt + futures_available
      reserve_total = total_available * 0.10
      max_spend = max(0.0, total_available - reserve_total)
      if funds_with_fee > max_spend:
        return {
          "rejected": True,
          "reason": "Exceeds spendable USDT after 10% reserve (incl fee)",
          "maxSpend": max_spend,
          "spotAvailable": spot_usdt,
          "futuresAvailable": futures_available,
          "feeRate": fee_rate,
          "hint": f"Retry with funds <= {max_spend:.2f}, or use futures with leverage instead, or sell a spot position to free capital.",
        }

      # If spot alone is insufficient but futures/financial has free balance, auto-transfer the shortfall.
      spot_reserve = spot_usdt * 0.10
      spot_spendable = max(0.0, spot_usdt - spot_reserve)
      
      if funds_with_fee > spot_spendable and not snapshot.paper_trading:
        need = funds_with_fee - spot_spendable
        
        # Try pulling from Financial (Earn) account first
        financial_available = 0.0
        if snapshot.financial_accounts:
          financial_totals = _aggregate_account_totals(snapshot.financial_accounts)
          financial_available = financial_totals.get("USDT", {}).get("available", 0.0)
        
        if financial_available > 0:
          financial_reserve = financial_available * 0.10
          financial_spendable = max(0.0, financial_available - financial_reserve)
          transfer_amt = min(need, financial_spendable)
          if transfer_amt > 0:
            try:
              transfer_used = kucoin.transfer_funds(
                currency="USDT",
                amount=transfer_amt,
                from_account="financial",
                to_account="trade",
              )
              spot_usdt += transfer_amt
              spot_spendable = max(0.0, spot_usdt - spot_usdt * 0.10)
              need = max(0, funds_with_fee - spot_spendable)
            except Exception as exc:
              logger.warning("Financial->spot transfer failed: %s", exc)

        # If still need more, try pulling from Futures account
        if need > 0 and futures_available > 0 and kucoin_futures:
          futures_reserve = futures_available * 0.10
          futures_spendable = max(0.0, futures_available - futures_reserve)
          transfer_amt = min(need, futures_spendable)
          if transfer_amt > 0:
            try:
              transfer_used = kucoin.transfer_funds(
                currency="USDT",
                amount=transfer_amt,
                from_account="contract",
                to_account="trade",
              )
              spot_usdt += transfer_amt
              spot_spendable = max(0.0, spot_usdt - spot_usdt * 0.10)
            except Exception as exc:
              logger.warning("Futures->spot transfer failed: %s", exc)

      # Re-check actual balance after transfers instead of trusting optimistic addition.
      if transfer_used:
        post_balances = kucoin.get_trade_accounts()
        spot_usdt = sum(float(b.available or 0) for b in post_balances if b.currency == "USDT")
        spot_spendable = max(0.0, spot_usdt - spot_usdt * 0.10)
        if funds_with_fee > spot_spendable:
          return {
            "rejected": True,
            "reason": "Insufficient balance after transfer (post-transfer verification)",
            "spotAvailable": spot_usdt,
            "fundsNeeded": funds_with_fee,
            "transferAttempt": transfer_used,
          }
    else:
      # No spend limits/transfers are needed for sells; they're bounded by position size below.
      max_spend = float("inf")

    price = mark_price
    size_est = funds_val / price if price else 0.0

    # Fetch symbol info for size/price increments so orders conform to exchange rules.
    try:
      sym_info = kucoin.get_symbol_info(symbol)
    except Exception as exc:
      logger.warning("Failed to fetch symbol info for %s: %s", symbol, exc)
      sym_info = {}
    base_increment = sym_info.get("baseIncrement", "0.00000001")

    planned_stop = None
    planned_tp = None
    rr_actual = None
    if (side or "").lower() == "buy":
      size_est = funds_val / price if price else None
      stop_val = None
      tp_val = None
      try:
        stop_val = float(stop_price) if stop_price is not None else None
      except Exception:
        stop_val = None
      try:
        tp_val = float(take_profit_price) if take_profit_price is not None else None
      except Exception:
        tp_val = None
      atr_mult = 2.0 if atr_multiple is None else float(atr_multiple)
      target_rr_val = 2.0 if target_rr is None else float(target_rr)
      if stop_val is None or tp_val is None:
        plan = await plan_spot_position(
          symbol=symbol,
          risk_pct=risk_pct,
          atr_multiple=atr_mult,
          target_rr=target_rr_val,
          entry_price=price if price else None,
        )
        if plan.get("error"):
          return {"rejected": True, "reason": "Plan failed", "plan": plan}
        stop_val = stop_val or plan.get("stopPrice")
        tp_val = tp_val or plan.get("targetPrice")
        size_est = size_est or plan.get("size")
      if not stop_val or not tp_val:
        return {"rejected": True, "reason": "Stop/TP required for buy", "stop": stop_val, "tp": tp_val}
      ok, reason = _stop_distance_ok(symbol, "sell", float(stop_val), price, fee_rate)
      if not ok:
        return {"rejected": True, "reason": reason, "minStopPct": max(0.003, 3 * fee_rate)}
      if tp_val <= price:
        return {"rejected": True, "reason": "Take-profit must be above entry"}
      rr_actual = (tp_val - price) / (price - float(stop_val)) if price > float(stop_val) else None
      if rr_actual is not None and rr_actual < 1.0:
        return {"rejected": True, "reason": "RR below minimum", "rr": rr_actual, "minRr": 1.0}
      planned_stop = float(stop_val)
      planned_tp = float(tp_val)
      size_est = float(size_est or 0.0)
      # PnL = (Exit - Entry) - Fees - Slippage
      expected_tp_pnl = (planned_tp * (1 - fee_rate - slippage_rate) - price * (1 + fee_rate + slippage_rate)) * size_est
      entry_cost_est = price * size_est * (1 + fee_rate + slippage_rate)
      expected_roi = expected_tp_pnl / entry_cost_est if entry_cost_est > 0 else 0.0
      min_profit_usd = cfg.trading.min_net_profit_usd
      min_roi = cfg.trading.min_profit_roi_pct
      if expected_tp_pnl < min_profit_usd and expected_roi < min_roi:
        min_tp_for_profit = price * (1 + fee_rate + slippage_rate) + min_profit_usd / size_est if size_est > 0 else None
        return {
          "rejected": True,
          "reason": f"Take-profit too close; requires at least ${min_profit_usd:.2f} net profit or {min_roi*100:.2f}% ROI (after slippage)",
          "expectedTpPnl": expected_tp_pnl,
          "expectedRoi": expected_roi,
          "minProfitUsd": min_profit_usd,
          "minRoiPct": min_roi,
          "size": size_est,
          "price": price,
          "tp": planned_tp,
          "feeRate": fee_rate,
          "slippageRate": slippage_rate,
          "hint": f"Set TP farther from entry (try >= {min_tp_for_profit:.2f} for buy) or increase position size to make the edge worthwhile. Alternatively, try futures with leverage to amplify the same move.",
        }

    # Adjust funds down so fee fits in spend cap.
    if funds_with_fee > max_spend and funds_val > 0:
      scale = max_spend / funds_with_fee
      funds_val = max(0.0, funds_val * scale)
      funds_with_fee = funds_val * (1 + fee_rate + slippage_rate)
      size_est = funds_val / price if price else 0.0

    order_req = KucoinOrderRequest(
      symbol=symbol,
      side="buy" if side == "buy" else "sell",  # enforce allowed values
      type="market",
      size=None,
      funds=f"{funds_val:.2f}" if side == "buy" else None,
      clientOid=str(uuid.uuid4()),
    )

    expected_pnl = None
    if side == "sell":
      roi_pct = None
      pos_info = _spot_position_info(symbol)
      if not pos_info:
        # Fallback: check live balance for externally-held coins
        fallback_size = _spot_position_size(symbol)
        if fallback_size > 0:
          pos_info = {"net": fallback_size, "avg_entry": None}
      net_size = pos_info["net"] if pos_info else 0.0
      if net_size <= 0:
        return {"rejected": True, "reason": "No spot position available to sell"}

      # Cancel active sell-side stop orders to release held balance before selling.
      if not snapshot.paper_trading:
        try:
          active_stops = kucoin.list_stop_orders(status="active", symbol=symbol) or []
          for order in active_stops:
            if not isinstance(order, dict):
              continue
            order_side = str(order.get("side") or "").lower()
            if order_side and order_side != "sell":
              continue
            oid = str(order.get("id") or order.get("orderId") or "").strip()
            coid = str(order.get("clientOid") or "").strip() or None
            if oid or coid:
              try:
                kucoin.cancel_stop_order(order_id=oid or None, client_oid=coid)
              except Exception as cancel_exc:
                logger.warning("Failed to cancel stop %s for %s: %s", oid or coid, symbol, cancel_exc)
        except Exception as exc:
          logger.warning("Failed to list stop orders for %s pre-sell: %s", symbol, exc)

        # Re-fetch balances after cancelling stops so freed holds are reflected.
        fresh_balances = kucoin.get_trade_accounts()
        fresh_by_currency = {}
        for bal in fresh_balances:
          fresh_by_currency[bal.currency] = fresh_by_currency.get(bal.currency, 0.0) + float(bal.available or 0)

      if size_est <= 0:
        size_est = net_size
        funds_val = size_est * price
      if size_est > net_size:
        size_est = net_size
        funds_val = size_est * price

      # Clamp to fresh available balance to avoid 200004 insufficient balance errors.
      base_currency = _base_currency(symbol)
      fresh_available = fresh_by_currency.get(base_currency, 0.0)
      if fresh_available <= 0:
        return {
          "rejected": True,
          "reason": "No available balance to sell (zero after stop cancellation)",
          "symbol": symbol,
          "staleNetSize": net_size,
        }
      if size_est > fresh_available:
        logger.warning("Clamping sell size for %s from %.8f to fresh available %.8f", symbol, size_est, fresh_available)
        size_est = fresh_available
        funds_val = size_est * price

      rationale_norm = (rationale or "").lower() if rationale else ""
      allow_loss_keywords = ("stop loss", "cut loss", "emergency", "liquidate", "portfolio review", "close position", "rebalance", "force sell", "force-sell", "supervisor")
      allow_loss = any(term in rationale_norm for term in allow_loss_keywords)

      if pos_info and pos_info.get("avg_entry"):
        # Known entry: full PnL validation
        breakeven_px = _fee_adjusted_breakeven(pos_info["avg_entry"], fee_rate + slippage_rate)
        expected_proceeds = price * size_est * (1 - fee_rate - slippage_rate)
        entry_cost = pos_info["avg_entry"] * size_est * (1 + fee_rate + slippage_rate)
        expected_pnl = expected_proceeds - entry_cost
        roi_pct = expected_pnl / entry_cost if entry_cost > 0 else 0.0

        if expected_pnl < 0 and not allow_loss:
          return {
            "rejected": True,
            "reason": f"Sell at a loss (expected PnL ${expected_pnl:.4f}); include 'stop loss', 'cut loss', or 'portfolio review' in rationale to allow",
            "breakevenPrice": breakeven_px,
            "expectedPnl": expected_pnl,
            "expectedRoi": roi_pct,
            "positionSize": net_size,
            "requestedSize": size_est,
            "feeRate": fee_rate,
            "hint": "To proceed, include 'cut loss', 'stop loss', or 'portfolio review' in the rationale, then retry.",
          }
        logger.info("Selling %s - Expected PnL: %.4f USD (ROI: %.2f%%)", symbol, expected_pnl, roi_pct * 100)
      else:
        # Unknown entry (externally acquired): allow sell with appropriate rationale
        if not allow_loss:
          return {
            "rejected": True,
            "reason": "Unknown entry price; include 'portfolio review', 'close position', or 'cut loss' in rationale to sell",
            "positionSize": net_size,
            "currentPrice": price,
            "hint": "Retry with rationale containing 'close position' or 'portfolio review'.",
          }
        logger.info("Selling %s (unknown entry) - Size: %.8f at price: %.6f", symbol, size_est, price)

      order_req.size = _truncate_to_increment(size_est, base_increment)
      order_req.funds = None
      truncated_size = float(order_req.size)
      base_min_size = _to_float(sym_info.get("baseMinSize"))
      quote_min_size = _to_float(sym_info.get("quoteMinSize"))
      if base_min_size > 0 and truncated_size < base_min_size:
        return {
          "rejected": True,
          "reason": "Sell size below exchange minimum after truncation",
          "truncatedSize": truncated_size,
          "baseMinSize": base_min_size,
          "positionSize": net_size,
        }
      if quote_min_size > 0 and truncated_size * price < quote_min_size:
        return {
          "rejected": True,
          "reason": "Sell notional below exchange minimum",
          "notional": truncated_size * price,
          "quoteMinSize": quote_min_size,
        }
    else:
      if size_est <= 0:
        return {"error": "Computed size is zero", "price": price}
    if snapshot.paper_trading:
      record = memory.record_trade(symbol, side, funds_val, paper=True, price=price, size=size_est)
      decision = None
      if confidence is not None:
        decision = memory.log_decision(
          symbol,
          f"spot_{side}",
          float(confidence),
          rationale or "paper trade",
          pnl=expected_pnl,
          paper=True,
        )
      bracket = {}
      if (side or "").lower() == "buy" and auto_protect and planned_stop and planned_tp:
        bracket = {
          "stop": {"paper": True, "stopPrice": planned_stop},
          "takeProfit": {"paper": True, "stopPrice": planned_tp},
        }
      return {
        "paper": True,
        "orderRequest": order_req.__dict__,
        "tradeRecord": record,
        "decisionLog": decision,
        "bracket": bracket,
        "rr": rr_actual,
      }
    # Limit order preference: use limit at best ask price instead of market for potential fee savings
    if cfg.trading.prefer_limit_orders and side == "buy" and price > 0:
      try:
        ticker = kucoin.get_ticker(symbol)
        best_ask = float(getattr(ticker, 'bestAsk', 0) or 0)
        if best_ask > 0:
          limit_price_str = str(best_ask)
          limit_size = _truncate_to_increment(funds_val / best_ask, base_increment)
          if float(limit_size) > 0:
            order_req = KucoinOrderRequest(
              symbol=symbol, side="buy", type="limit",
              size=limit_size, price=limit_price_str,
              clientOid=str(uuid.uuid4()),
            )
            price = best_ask
            size_est = float(limit_size)
      except Exception:
        pass

    try:
      res = kucoin.place_order(order_req).__dict__
      record = memory.record_trade(symbol, side, funds_val, paper=False, price=price, size=size_est)
      res["tradeRecord"] = record
      if side == "sell" and expected_pnl is not None:
        res["expectedPnl"] = expected_pnl
        res["expectedRoi"] = roi_pct
      if confidence is not None:
        res["decisionLog"] = memory.log_decision(
          symbol,
          f"spot_{side}",
          float(confidence),
          rationale or "live trade",
          pnl=expected_pnl,
          paper=False,
        )
      if (side or "").lower() == "buy" and auto_protect and planned_stop and planned_tp:
        # Cancel existing sell-side stops for this symbol to avoid stacking duplicates.
        try:
          existing_stops = kucoin.list_stop_orders(status="active", symbol=symbol) or []
          for old_order in existing_stops:
            if not isinstance(old_order, dict):
              continue
            if str(old_order.get("side") or "").lower() != "sell":
              continue
            oid = str(old_order.get("id") or old_order.get("orderId") or "").strip()
            coid = str(old_order.get("clientOid") or "").strip() or None
            if oid or coid:
              try:
                kucoin.cancel_stop_order(order_id=oid or None, client_oid=coid)
              except Exception:
                pass
        except Exception as exc:
          logger.warning("Failed to clean up old stops for %s before bracket: %s", symbol, exc)

        # Cover the full position (prior holdings + this buy) since we cancelled all old stops.
        prior_position = _spot_position_size(symbol)
        this_buy_size = size_est or (funds_val / price if price else 0)
        size_for_exit = prior_position + this_buy_size if prior_position > 0 else this_buy_size
        bracket: Dict[str, Any] = {}
        stop_res = await _place_spot_stop_order_impl(
          symbol=symbol,
          side="sell",
          stop_price=planned_stop,
          stop_price_type="MP",
          order_type="market",
          size=size_for_exit,
        )
        bracket["stop"] = stop_res
        tp_res = await _place_spot_stop_order_impl(
          symbol=symbol,
          side="sell",
          stop_price=planned_tp,
          stop_price_type="TP",
          order_type="limit",
          size=size_for_exit,
          limit_price=planned_tp,
        )
        bracket["takeProfit"] = tp_res
        res["bracket"] = bracket
        res["rr"] = rr_actual
      return res
    except Exception as exc:
      return {"error": str(exc), "orderRequest": order_req.__dict__, "transfer": transfer_used}

  @function_tool
  async def place_limit_order(
    symbol: str,
    side: str,
    funds: float,
    entry_price: float,
    confidence: float | None = None,
    rationale: str | None = None,
    stop_price: float | None = None,
    take_profit_price: float | None = None,
    risk_pct: float | None = None,
  ) -> Dict[str, Any]:
    """Place a spot limit entry order at a technically derived target price.
    Use for new entries where you want to wait for price to reach a key level
    (EMA, Bollinger Band, swing high/low, VWAP) rather than entering at the
    current market price. Rejects if entry_price is too close to current price
    — use place_market_order instead when price is already at the target level.
    Bracket TP/SL are placed on the NEXT run after the limit fills."""
    symbol = _normalize_symbol(symbol)
    if symbol not in allowed_symbols:
      return {"error": "Unsupported symbol", "allowed": sorted(allowed_symbols)}
    try:
      entry_price_val = float(entry_price)
    except (TypeError, ValueError):
      return {"error": "entry_price must be a valid number"}
    if entry_price_val <= 0:
      return {"error": "entry_price must be positive"}

    side_lower = (side or "").lower()

    if side_lower == "buy" and snapshot.trading_restricted:
      return {"rejected": True, "reason": f"Trading restricted (close-only mode): {snapshot.restriction_reason}", "hint": "Only sell/close operations are allowed during circuit breaker activation."}

    if side_lower == "buy" and cfg.trading.post_loss_cooldown_minutes > 0:
      last_loss_ts = memory.last_loss_time(symbol)
      if last_loss_ts:
        elapsed_min = (int(time.time()) - last_loss_ts) / 60
        if elapsed_min < cfg.trading.post_loss_cooldown_minutes:
          remaining = int(cfg.trading.post_loss_cooldown_minutes - elapsed_min)
          return {"rejected": True, "reason": f"Post-loss cooldown active for {symbol} ({remaining}min remaining)", "hint": "Try a different symbol or wait for cooldown to expire."}

    if side_lower == "buy" and cfg.trading.min_trade_interval_minutes > 0:
      last_trade_ts = memory.last_trade_time(symbol)
      if last_trade_ts:
        elapsed_min = (int(time.time()) - last_trade_ts) / 60
        if elapsed_min < cfg.trading.min_trade_interval_minutes:
          remaining = int(cfg.trading.min_trade_interval_minutes - elapsed_min)
          return {"rejected": True, "reason": f"Trade interval cooldown for {symbol} ({remaining}min remaining)", "hint": f"Minimum {cfg.trading.min_trade_interval_minutes:.0f}min between trades."}

    if side_lower == "buy" and symbol in _daily_gate_state:
      gate = _daily_gate_state[symbol]
      daily_exhausted_l = gate.get("daily_exhausted", False)
      daily_bias_raw_l = gate.get("daily_bias_raw", gate.get("daily_bias", "neutral"))
      if daily_exhausted_l and daily_bias_raw_l == "bullish":
        logger.warning("ANTI-FOMO BLOCK: spot limit buy %s rejected — daily bullish exhausted", symbol)
        return {"rejected": True, "reason": "Daily exhaustion: bullish trend overextended — no continuation entry", "hint": "Daily RSI is at an extreme high. Wait for the pullback or trade counter-trend."}
      if gate.get("daily_bias") == "bearish" and not daily_exhausted_l:
        return {"rejected": True, "reason": "Daily gate: 1D trend is bearish — spot buy blocked", "hint": "Trade with the daily trend or switch symbol."}
      intraday_bias_1h_l = gate.get("intraday_bias_1h", "neutral")
      if intraday_bias_1h_l == "bearish":
        logger.warning("1H ALIGN BLOCK: spot limit buy %s rejected — 1h bias is bearish", symbol)
        return {"rejected": True, "reason": "1h trend is bearish — long entry blocked", "hint": "1h timeframe is in a downtrend. The daily uptrend is in correction. Wait for 1h to turn neutral/bullish."}
      tf_conflict_l = gate.get("timeframe_conflict", False)
      intraday_bias_15m_l = gate.get("intraday_bias_15m", "neutral")
      if tf_conflict_l and intraday_bias_15m_l == "bearish":
        logger.warning("TF CONFLICT BLOCK: spot limit buy %s rejected — daily/intraday split, 15m bearish opposes buy", symbol)
        return {"rejected": True, "reason": "Timeframe conflict: 15m bearish opposes proposed buy", "hint": "Wait for 15m to align with the higher-TF bias, or pick a different symbol."}

    _atr_scale_l = 1.0
    if side_lower == "buy" and symbol in _daily_gate_state:
      gate = _daily_gate_state[symbol]
      daily_atr_pct = gate.get("daily_atr_pct")
      if daily_atr_pct is not None and daily_atr_pct > cfg.trading.max_atr_pct_for_entry:
        hard_limit = cfg.trading.max_atr_pct_for_entry * 1.5
        if daily_atr_pct > hard_limit:
          return {"rejected": True, "reason": f"Extreme volatility: ATR={daily_atr_pct:.2f}% > {hard_limit:.1f}% hard limit"}
        _atr_scale_l = max(0.30, (cfg.trading.max_atr_pct_for_entry / daily_atr_pct) ** 2)
        logger.info("VOLATILITY SOFT GATE: spot limit %s ATR=%.2f%% — scaling position to %.0f%%", symbol, daily_atr_pct, _atr_scale_l * 100)

    if side_lower == "buy" and confidence is not None and confidence < cfg.trading.min_confidence:
      return {"rejected": True, "reason": f"Confidence {confidence:.2f} below minimum {cfg.trading.min_confidence}"}

    # Correlation gate: block alt spot BUYS while BTC's daily regime is bearish (high-beta blowup guard).
    if block_alt_long_in_btc_downtrend(symbol=symbol, side=side_lower, btc_daily_bias=_btc_daily_bias(), cfg=cfg.regime):
      logger.warning("CORRELATION GATE BLOCK: spot limit buy %s rejected — BTC daily regime is bearish", symbol)
      return {"rejected": True, "reason": f"Correlation gate: BTC daily regime is bearish — spot buy on alt {symbol} blocked", "hint": "Alts are high-beta to BTC; do NOT buy an altcoin while BTC's daily trend is down. Wait for BTC's daily to turn or trade a major."}

    try:
      funds_val = float(funds or 0)
    except (TypeError, ValueError):
      funds_val = 0.0
    funds_val = funds_val * _atr_scale_l
    if funds_val <= 0 or not symbol:
      return {"error": "Invalid funds or symbol"}
    if funds_val > snapshot.max_position_usd:
      return {"rejected": True, "reason": "Exceeds maxPositionUsd"}
    trades_today = memory.trades_today(symbol)
    if trades_today >= cfg.trading.max_trades_per_symbol_per_day:
      return {"rejected": True, "reason": "Daily trade cap reached"}

    if cfg.trading.sentiment_filter_enabled and side_lower == "buy":
      latest_sent = memory.latest_sentiment(symbol)
      day_key = int(time.time() // 86400)
      if not latest_sent or latest_sent.get("day") != day_key:
        return {"rejected": True, "reason": "Sentiment missing for today"}
      if latest_sent.get("score", 0) < cfg.trading.sentiment_min_score:
        return {"rejected": True, "reason": "Sentiment below threshold", "score": latest_sent.get("score")}

    try:
      current_price = float(snapshot.tickers[symbol].price)
    except Exception:
      return {"error": "Unable to fetch current price for deviation check"}
    if current_price <= 0:
      return {"error": "Invalid current price"}

    deviation = abs(entry_price_val - current_price) / current_price
    if deviation < cfg.trading.min_entry_deviation_pct:
      return {
        "rejected": True,
        "reason": f"entry_price {entry_price_val} is within {deviation*100:.3f}% of current price {current_price:.4f} (minimum deviation: {cfg.trading.min_entry_deviation_pct*100:.2f}%)",
        "hint": "Price is already at your target — use place_market_order to enter immediately, or pick a further entry_price to genuinely wait for a better level.",
        "currentPrice": current_price,
        "entryPrice": entry_price_val,
      }

    try:
      sym_info = kucoin.get_symbol_info(symbol)
    except Exception:
      sym_info = {}
    base_increment = sym_info.get("baseIncrement", "0.00000001")
    size_str = _truncate_to_increment(funds_val / entry_price_val, base_increment)
    size_val = float(size_str)
    if size_val <= 0:
      return {"error": "Computed size is zero; increase funds or adjust entry_price"}
    base_min_size = _to_float(sym_info.get("baseMinSize"))
    if base_min_size > 0 and size_val < base_min_size:
      return {"rejected": True, "reason": "Size below exchange minimum", "size": size_val, "minSize": base_min_size, "hint": "Increase funds or choose a symbol with smaller minimum lot size."}

    order_req = KucoinOrderRequest(
      symbol=symbol,
      side=side_lower,
      type="limit",
      size=size_str,
      price=str(entry_price_val),
      clientOid=str(uuid.uuid4()),
    )
    expiry_ts = int(time.time()) + int(cfg.trading.entry_limit_expiry_minutes * 60)
    pending_protection = {"stopPrice": stop_price, "takeProfitPrice": take_profit_price}
    note = (
      f"Limit {side_lower} placed at {entry_price_val} (current {current_price:.4f}, {deviation*100:.2f}% away). "
      f"Order expires in {cfg.trading.entry_limit_expiry_minutes:.0f}min if unfilled. "
      "On next run: check open positions — if filled, place bracket TP/SL with place_spot_stop_order."
    )

    if snapshot.paper_trading:
      record = memory.record_trade(symbol, side, funds_val, paper=True, price=entry_price_val, size=size_val)
      decision = None
      if confidence is not None:
        decision = memory.log_decision(symbol, f"spot_{side_lower}_limit", float(confidence), rationale or "paper limit entry", paper=True)
      return {
        "paper": True,
        "pendingLimitEntry": True,
        "orderRequest": order_req.__dict__,
        "entryPrice": entry_price_val,
        "currentPrice": current_price,
        "deviationPct": round(deviation * 100, 3),
        "expiresAt": expiry_ts,
        "pendingProtection": pending_protection,
        "tradeRecord": record,
        "decisionLog": decision,
        "note": note,
      }

    try:
      res = kucoin.place_order(order_req).__dict__
      if confidence is not None:
        res["decisionLog"] = memory.log_decision(symbol, f"spot_{side_lower}_limit", float(confidence), rationale or "live limit entry", paper=False)
      res["pendingLimitEntry"] = True
      res["entryPrice"] = entry_price_val
      res["currentPrice"] = current_price
      res["deviationPct"] = round(deviation * 100, 3)
      res["expiresAt"] = expiry_ts
      res["pendingProtection"] = pending_protection
      res["note"] = note
      return res
    except Exception as exc:
      return {"error": str(exc), "orderRequest": order_req.__dict__}

  async def _place_spot_stop_order_impl(
    symbol: str,
    side: str,
    stop_price: float,
    stop_price_type: str = "MP",
    order_type: str = "limit",
    size: float | None = None,
    funds: float | None = None,
    limit_price: float | None = None,
    client_oid: str | None = None,
  ) -> Dict[str, Any]:
    """Place a spot stop order (stop-loss or take-profit). stop_price_type: TP(trigger when price rises) or MP(falls)."""
    symbol = _normalize_symbol(symbol)
    if symbol not in allowed_symbols:
      return {"error": "Unsupported symbol", "allowed": sorted(allowed_symbols)}
    try:
      stop_price_f = float(stop_price or 0)
    except Exception:
      return {"error": "Invalid stop price"}
    if stop_price_f <= 0:
      return {"error": "Stop price must be positive"}
    ref_px = _to_float(snapshot.tickers[symbol].price if symbol in snapshot.tickers else None)
    validation_side = side
    if (side or "").lower() == "sell" and stop_price_type.upper() == "TP":
      validation_side = "buy"  # TP on a long should be above price
    ok, reason = _stop_distance_ok(symbol, validation_side, stop_price_f, ref_px, fees.get("spot_taker", 0.001))
    if not ok:
      return {"rejected": True, "reason": reason, "price": ref_px, "stop": stop_price_f}
    # Fetch symbol info to respect baseIncrement for size.
    try:
      sym_info = kucoin.get_symbol_info(symbol)
    except Exception:
      sym_info = {}
    base_inc = sym_info.get("baseIncrement", "0.00000001")

    order_req = KucoinOrderRequest(
      symbol=symbol,
      side="buy" if side == "buy" else "sell",
      type="limit" if order_type == "limit" else "market",
      size=_truncate_to_increment(float(size), base_inc) if size is not None else None,
      funds=f"{funds:.2f}" if funds is not None else None,
      price=f"{limit_price:.6f}" if limit_price is not None else None,
      clientOid=client_oid or str(uuid.uuid4()),
      stopPrice=f"{stop_price_f}",
      stopPriceType="TP" if stop_price_type.upper() == "TP" else "MP",
    )
    if snapshot.paper_trading:
      return {"paper": True, "orderRequest": order_req.__dict__}
    try:
      res = kucoin.place_stop_order(order_req).__dict__
      res["orderRequest"] = order_req.__dict__
      return res
    except Exception as exc:
      return {"error": str(exc), "orderRequest": order_req.__dict__}

  @function_tool
  async def place_spot_stop_order(
    symbol: str,
    side: str,
    stop_price: float,
    stop_price_type: str = "MP",
    order_type: str = "limit",
    size: float | None = None,
    funds: float | None = None,
    limit_price: float | None = None,
    client_oid: str | None = None,
  ) -> Dict[str, Any]:
    """Place a spot stop order (stop-loss or take-profit). stop_price_type: TP(trigger when price rises) or MP(falls)."""
    return await _place_spot_stop_order_impl(
      symbol=symbol,
      side=side,
      stop_price=stop_price,
      stop_price_type=stop_price_type,
      order_type=order_type,
      size=size,
      funds=funds,
      limit_price=limit_price,
      client_oid=client_oid,
    )

  @function_tool
  async def cancel_spot_stop_order(order_id: str | None = None, client_oid: str | None = None) -> Dict[str, Any]:
    """Cancel a spot stop order by orderId or clientOid."""
    if snapshot.paper_trading:
      return {"paper": True, "cancelled": {"orderId": order_id, "clientOid": client_oid}}
    try:
      res = kucoin.cancel_stop_order(order_id=order_id, client_oid=client_oid)
      return {"cancelled": res, "orderId": order_id, "clientOid": client_oid}
    except Exception as exc:
      return {"error": str(exc), "orderId": order_id, "clientOid": client_oid}

  @function_tool
  async def cancel_spot_limit_order(order_id: str) -> Dict[str, Any]:
    """Cancel a pending spot limit order (non-stop) by orderId."""
    if snapshot.paper_trading:
      return {"paper": True, "cancelled": {"orderId": order_id}}
    try:
      res = kucoin.cancel_order(order_id)
      return {"cancelled": res, "orderId": order_id}
    except Exception as exc:
      return {"error": str(exc), "orderId": order_id}

  @function_tool
  async def list_spot_stop_orders(status: str = "active", symbol: str | None = None) -> Dict[str, Any]:
    """List spot stop orders; status usually 'active' or 'done'."""
    try:
      orders = kucoin.list_stop_orders(status=status, symbol=symbol)
      return {"orders": orders, "status": status, "symbol": symbol}
    except Exception as exc:
      return {"error": str(exc), "status": status, "symbol": symbol}

  @function_tool
  async def set_spot_position_protection(
    symbol: str,
    take_profit_price: float | None = None,
    stop_loss_price: float | None = None,
    cancel_existing: bool = True,
  ) -> Dict[str, Any]:
    """Add or replace TP/SL for an existing open spot position using sell stop orders."""
    symbol = _normalize_symbol(symbol)
    if symbol not in allowed_symbols:
      return {"error": "Unsupported symbol", "allowed": sorted(allowed_symbols)}

    try:
      tp_val = float(take_profit_price) if take_profit_price is not None else None
    except Exception:
      tp_val = None
    try:
      sl_val = float(stop_loss_price) if stop_loss_price is not None else None
    except Exception:
      sl_val = None
    if not tp_val and not sl_val:
      return {"error": "At least one of take_profit_price or stop_loss_price is required"}

    position_size = _spot_position_size(symbol)
    if position_size <= 0:
      return {"error": "No open spot position", "symbol": symbol}

    cancelled: list[Dict[str, Any]] = []
    cancel_errors: list[Dict[str, Any]] = []
    active_orders: list[Dict[str, Any]] = []
    if cancel_existing:
      try:
        active_orders = kucoin.list_stop_orders(status="active", symbol=symbol) or []
      except Exception as exc:
        cancel_errors.append({"stage": "list", "error": str(exc)})
      for order in active_orders:
        if not isinstance(order, dict):
          continue
        order_side = str(order.get("side") or "").lower()
        if order_side and order_side != "sell":
          continue
        order_symbol = _normalize_symbol(str(order.get("symbol") or symbol))
        if order_symbol and order_symbol != symbol:
          continue
        order_id = str(order.get("id") or order.get("orderId") or "").strip()
        client_oid = str(order.get("clientOid") or "").strip() or None
        if not order_id and not client_oid:
          continue
        if snapshot.paper_trading:
          cancelled.append({"paper": True, "orderId": order_id or None, "clientOid": client_oid, "symbol": symbol})
          continue
        try:
          res = kucoin.cancel_stop_order(order_id=order_id or None, client_oid=client_oid)
          cancelled.append({"orderId": order_id or None, "clientOid": client_oid, "response": res})
        except Exception as exc:
          cancel_errors.append({"orderId": order_id or None, "clientOid": client_oid, "error": str(exc)})

    bracket: Dict[str, Any] = {}
    if sl_val:
      bracket["stopLoss"] = await _place_spot_stop_order_impl(
        symbol=symbol,
        side="sell",
        stop_price=sl_val,
        stop_price_type="MP",
        order_type="market",
        size=position_size,
      )
    if tp_val:
      if cfg.trading.partial_tp_enabled and position_size > 0:
        try:
          sym_info = kucoin.get_symbol_info(symbol)
        except Exception:
          sym_info = {}
        base_increment = sym_info.get("baseIncrement", "0.00000001")
        base_min_size = _to_float(sym_info.get("baseMinSize"))
        cur_price = float(snapshot.tickers.get(symbol, snapshot.tickers.get(list(snapshot.tickers.keys())[0])).price) if snapshot.tickers else 0.0
        tp_distance = tp_val - cur_price if cur_price > 0 else 0
        tp1_price = cur_price + tp_distance * 0.6 if tp_distance > 0 else tp_val
        tp2_price = tp_val
        size_tranche1 = float(_truncate_to_increment(position_size * 0.6, base_increment))
        size_tranche2 = float(_truncate_to_increment(position_size - size_tranche1, base_increment))
        tranches_placed = []
        if size_tranche1 >= base_min_size and tp1_price > cur_price:
          res1 = await _place_spot_stop_order_impl(
            symbol=symbol, side="sell", stop_price=tp1_price,
            stop_price_type="TP", order_type="limit", size=size_tranche1, limit_price=tp1_price,
          )
          tranches_placed.append({"tranche": 1, "size": size_tranche1, "price": tp1_price, "result": res1})
        if size_tranche2 >= base_min_size and tp2_price > cur_price:
          res2 = await _place_spot_stop_order_impl(
            symbol=symbol, side="sell", stop_price=tp2_price,
            stop_price_type="TP", order_type="limit", size=size_tranche2, limit_price=tp2_price,
          )
          tranches_placed.append({"tranche": 2, "size": size_tranche2, "price": tp2_price, "result": res2})
        if tranches_placed:
          bracket["takeProfit"] = {"staged": True, "tranches": tranches_placed}
        else:
          bracket["takeProfit"] = await _place_spot_stop_order_impl(
            symbol=symbol, side="sell", stop_price=tp_val,
            stop_price_type="TP", order_type="limit", size=position_size, limit_price=tp_val,
          )
      else:
        bracket["takeProfit"] = await _place_spot_stop_order_impl(
          symbol=symbol,
          side="sell",
          stop_price=tp_val,
          stop_price_type="TP",
          order_type="limit",
          size=position_size,
          limit_price=tp_val,
        )

    return {
      "symbol": symbol,
      "positionSize": position_size,
      "cancelledOrders": cancelled,
      "cancelErrors": cancel_errors,
      "bracket": bracket,
    }


  # ── Decision ────────────────────────────────────────────────────────────────────────────────────
  @function_tool
  async def decline_trade(reason: str, confidence: float) -> Dict[str, Any]:
    """Decline trading due to low confidence or risk."""
    memory.log_decision("ALL", "decline", confidence, reason, paper=True)
    return {"skipped": True, "reason": reason, "confidence": confidence}


  # ── Market data & multi-timeframe analysis ──────────────────────────────────────────────────────
  @function_tool
  async def fetch_recent_candles(
    symbol: str,
    interval: str = "1min",
    lookback_minutes: int = 120,
  ) -> Dict[str, Any]:
    """Fetch recent candles for symbol. Interval options: 1min, 5min, 15min, 1hour. Caps to 500 rows."""
    symbol = _normalize_symbol(symbol)

    interval_seconds = {"1min": 60, "5min": 300, "15min": 900, "1hour": 3600}
    if interval not in interval_seconds:
      return {"error": "Invalid interval", "allowed": list(interval_seconds.keys())}

    lookback_min = max(1, min(int(lookback_minutes or 0), 720))
    end_at = int(time.time())
    # bound number of points to avoid oversized responses
    max_points = 500
    interval_sec = interval_seconds[interval]
    points = min(max_points, max(1, int(lookback_min * 60 / interval_sec)))
    start_at = end_at - points * interval_sec

    candles = kucoin.get_candles(
      symbol,
      interval=interval,
      start_at=start_at,
      end_at=end_at,
    )

    return {
      "symbol": symbol,
      "interval": interval,
      "startAt": start_at,
      "endAt": end_at,
      "points": candles[:max_points],
      "rows": len(candles),
    }

  @function_tool
  async def fetch_orderbook(symbol: str, depth: int = 20) -> Dict[str, Any]:
    """Fetch level2 orderbook snapshot (depth 20 or 100)."""
    symbol = _normalize_symbol(symbol)
    depth_safe = 20 if depth <= 20 else 100
    ob = kucoin.get_orderbook_levels(symbol, depth=depth_safe)
    # trim in case API returns more than requested
    ob["bids"] = ob.get("bids", [])[:depth_safe]
    ob["asks"] = ob.get("asks", [])[:depth_safe]
    return {"symbol": symbol, "depth": depth_safe, "orderbook": ob}

  @function_tool
  async def analyze_market_context(
    symbol: str,
    fast_interval: str = "15min",
    slow_interval: str = "1hour",
    lookback_minutes: int = 360,
  ) -> Dict[str, Any]:
    """Compute EMA/RSI/MACD/ATR/Bollinger/VWAP/VolumeProfile across up to four intervals (15m, 1h, 4h, 1d) and summarize bias. The 1D acts as a regime gate — it can veto counter-trend trades. When futures are enabled, also returns funding rate, open interest, basis, OI-price signal, and funding divergence."""
    symbol = _normalize_symbol(symbol)

    interval_order: list[str] = []
    for iv in [fast_interval, slow_interval]:
      if iv not in INTERVAL_SECONDS:
        return {"error": "Invalid interval", "allowed": list(INTERVAL_SECONDS.keys())}
      if iv not in interval_order:
        interval_order.append(iv)
    for extra in ("4hour", "1day"):
      if extra not in interval_order:
        interval_order.append(extra)

    snapshots: list[Dict[str, Any]] = []
    end_at = int(time.time())
    for iv in interval_order:
      interval_sec = INTERVAL_SECONDS[iv]
      if iv == "1day":
        lookback_min = 43200  # 30 days
      elif iv == "4hour":
        lookback_min = 2880
      else:
        lookback_min = max(120, min(int(lookback_minutes or 0), 720))
      points = min(500, max(50, int(lookback_min * 60 / interval_sec)))
      start_at = end_at - points * interval_sec
      candles = kucoin.get_candles(symbol, interval=iv, start_at=start_at, end_at=end_at)
      if not candles:
        if iv in ("4hour", "1day"):
          continue
        return {"error": "No candles returned", "interval": iv}
      try:
        df = candles_to_dataframe(candles)
        snapshot = summarize_interval(df, iv)
        snapshot["rows"] = len(df)
        snapshots.append(snapshot)
      except Exception as exc:
        if iv in ("4hour", "1day"):
          continue
        return {"error": str(exc), "interval": iv}

    summary = summarize_multi_timeframe(snapshots)
    # Extract daily ATR% and 15m ATR% for volatility filtering
    daily_atr_pct = None
    intraday_atr_pct = None
    for snap in snapshots:
      iv = snap.get("interval")
      atr_pct = snap.get("atr_pct")
      if iv == "1day" and atr_pct is not None:
        daily_atr_pct = atr_pct
      elif iv == "15min" and atr_pct is not None:
        intraday_atr_pct = atr_pct
    _daily_gate_state[symbol] = {
      "daily_bias": summary.get("daily_bias", "neutral"),
      "daily_bias_raw": summary.get("daily_bias_raw", summary.get("daily_bias", "neutral")),
      "daily_gate_applied": summary.get("daily_gate_applied", False),
      "daily_exhausted": summary.get("daily_exhausted", False),
      "overall_bias": summary.get("overall_bias", "neutral"),
      "daily_atr_pct": daily_atr_pct,
      "intraday_atr_pct": intraday_atr_pct,
      "squeeze_breakout": summary.get("squeeze_breakout"),
      "timeframe_conflict": summary.get("timeframe_conflict", False),
      "intraday_bias_15m": summary.get("intraday_bias_15m", "neutral"),
      "intraday_bias_1h": summary.get("intraday_bias_1h", "neutral"),
    }
    result: Dict[str, Any] = {"symbol": symbol, "snapshots": snapshots, "summary": summary}

    if cfg.kucoin_futures.enabled and kucoin_futures:
      fsym = _to_futures_symbol(symbol)
      if fsym:
        futures_data: Dict[str, Any] = {"symbol": fsym}
        current_funding_rate = None
        try:
          fr = kucoin_futures.get_funding_rate(fsym)
          current_funding_rate = fr.get("value")
          futures_data["fundingRate"] = current_funding_rate
          futures_data["predictedRate"] = fr.get("predictedValue")
        except Exception:
          pass
        current_oi = None
        try:
          contract = kucoin_futures.get_contract_detail(fsym)
          current_oi = contract.get("openInterest")
          futures_data["openInterest"] = current_oi
          futures_data["volumeOf24h"] = contract.get("volumeOf24h")
          futures_data["turnoverOf24h"] = contract.get("turnoverOf24h")
        except Exception:
          pass
        try:
          mp = kucoin_futures.get_mark_price(fsym)
          mark = float(mp.get("value") or 0)
          index = float(mp.get("indexPrice") or 0)
          futures_data["markPrice"] = mark
          futures_data["indexPrice"] = index
          if mark and index:
            basis = mark - index
            futures_data["basis"] = round(basis, 6)
            futures_data["basisPct"] = round(basis / index * 100, 4)
        except Exception:
          pass

        # OI-Price divergence signal
        price_dir = None
        for snap in snapshots:
          if snap.get("interval") in ("4hour", "1hour"):
            ema_f = snap.get("ema_fast")
            ema_s = snap.get("ema_slow")
            if ema_f is not None and ema_s is not None:
              price_dir = "up" if ema_f > ema_s else "down"
              break
        if price_dir and current_oi is not None:
          try:
            oi_val = float(current_oi)
            vol_24h = float(futures_data.get("volumeOf24h") or 0)
            oi_trend = "up" if vol_24h > 0 else "neutral"
            if oi_trend != "neutral":
              if price_dir == "up" and oi_trend == "up":
                futures_data["oiPriceSignal"] = "strong_trend"
                futures_data["oiPriceHint"] = "Rising price + rising OI = strong trend continuation. Stay in or add to position."
              elif price_dir == "up" and oi_trend != "up":
                futures_data["oiPriceSignal"] = "short_covering"
                futures_data["oiPriceHint"] = "Rising price + flat/falling OI = short covering rally. Exit longs cautiously, don't add."
              elif price_dir == "down" and oi_trend == "up":
                futures_data["oiPriceSignal"] = "aggressive_shorts"
                futures_data["oiPriceHint"] = (
                  "Falling price + rising OI = aggressive short BUILDING. Trend likely continues lower. "
                  "Do NOT enter contrarian longs hoping for a squeeze unless price reclaims a clear structural level "
                  "(prior swing high or daily EMA cross). Stay short, exit longs, or stand aside."
                )
              else:
                futures_data["oiPriceSignal"] = "long_capitulation"
                futures_data["oiPriceHint"] = "Falling price + flat/falling OI = long capitulation. Potential reversal zone for contrarian long."
          except (TypeError, ValueError):
            pass

        # Funding rate divergence
        if current_funding_rate is not None and price_dir:
          try:
            fr_val = float(current_funding_rate)
            if price_dir == "up" and fr_val < 0:
              futures_data["fundingDivergence"] = "hidden_strength"
              futures_data["fundingDivergenceHint"] = "Price rising but funding negative = hidden bullish strength (shorts paying longs)."
            elif price_dir == "down" and fr_val > 0:
              futures_data["fundingDivergence"] = "hidden_weakness"
              futures_data["fundingDivergenceHint"] = "Price falling but funding positive = hidden bearish weakness (longs paying shorts)."
            elif (price_dir == "up" and fr_val > 0) or (price_dir == "down" and fr_val < 0):
              futures_data["fundingDivergence"] = "aligned"
              futures_data["fundingDivergenceHint"] = "Funding rate aligned with price direction."
            else:
              futures_data["fundingDivergence"] = "neutral"
          except (TypeError, ValueError):
            pass

        if futures_data.keys() - {"symbol"}:
          result["futures"] = futures_data

    return result

  @function_tool
  async def plan_spot_position(
    symbol: str,
    risk_pct: float | None = None,
    atr_multiple: float = 2.0,
    target_rr: float = 2.0,
    entry_price: float | None = None,
  ) -> Dict[str, Any]:
    """Size a spot trade using risk-per-trade % with ATR-based stop/target. Uses Kelly criterion when enabled and sufficient trade history exists."""
    symbol = _normalize_symbol(symbol)
    if symbol not in allowed_symbols:
      return {"error": "Unsupported symbol", "allowed": sorted(allowed_symbols)}

    balance_usdt = balances_by_currency.get("USDT", 0.0)
    if balance_usdt <= 0:
      return {"error": "No USDT balance available"}
    risk_fraction = risk_pct if risk_pct is not None else cfg.trading.risk_per_trade_pct

    # Kelly criterion: dynamically adjust risk fraction based on historical edge
    kelly_used = False
    if risk_pct is None and cfg.trading.kelly_sizing_enabled:
      perf = memory.performance_summary()
      total_closed = perf.get("closedWithPnl", 0)
      if total_closed >= cfg.trading.kelly_min_trades:
        kelly_frac = memory.kelly_fraction(venue="spot")
        risk_fraction = kelly_frac
        kelly_used = True

    risk_fraction = max(0.0, float(risk_fraction))
    risk_dollars = balance_usdt * risk_fraction
    if risk_dollars <= 0:
      return {"error": "Risk dollars computed to zero", "riskPct": risk_fraction}

    fee_rate = fees.get("spot_taker", 0.001)
    spendable = max(0.0, balance_usdt * 0.90)
    price = float(entry_price) if entry_price else float(snapshot.tickers[symbol].price)

    lookback_min = 180
    end_at = int(time.time())
    interval = "15min"
    interval_sec = INTERVAL_SECONDS[interval]
    points = min(500, max(50, int(lookback_min * 60 / interval_sec)))
    start_at = end_at - points * interval_sec
    candles = kucoin.get_candles(symbol, interval=interval, start_at=start_at, end_at=end_at)
    if not candles:
      return {"error": "No candles returned for ATR sizing", "interval": interval}
    df = candles_to_dataframe(candles)
    enriched = compute_indicators(df)
    atr_raw = enriched["atr"].iloc[-1]
    try:
      atr_val = float(atr_raw)
    except Exception:
      atr_val = None
    if atr_val != atr_val:  # NaN check
      atr_val = None
    stop_distance = atr_val * atr_multiple if atr_val else price * 0.015  # fallback 1.5%
    if stop_distance <= 0:
      return {"error": "Stop distance invalid", "atr": atr_val}

    raw_size = risk_dollars / stop_distance
    notional_unclipped = raw_size * price
    cap_notional = min(snapshot.max_position_usd, spendable)
    notional = min(notional_unclipped, cap_notional)
    notional_with_fee = notional * (1 + fee_rate)
    size = notional / price if price else 0
    stop_price = max(0.0, price - stop_distance)

    daily_atr_pct = _daily_gate_state.get(symbol, {}).get("daily_atr_pct")
    if daily_atr_pct is not None and daily_atr_pct >= 4.0:
      vol_rr_multiplier = min(2.0, 1.0 + (daily_atr_pct - 4.0) / 6.0)
    else:
      vol_rr_multiplier = 1.0
    effective_rr = target_rr * vol_rr_multiplier
    target_price = price + stop_distance * effective_rr

    trades_today = memory.trades_today(symbol)

    warnings: list[str] = []
    if atr_val and atr_val / price * 100 >= 5:
      warnings.append("Volatility elevated (ATR% >=5); consider smaller size or skip.")
    if notional_unclipped > cap_notional:
      warnings.append("Size clipped by maxPositionUsd or 10% reserve.")
    if trades_today >= cfg.trading.max_trades_per_symbol_per_day:
      warnings.append("Trade cap reached; do not enter without manual override.")

    return {
      "symbol": symbol,
      "price": price,
      "riskPct": risk_fraction,
      "riskDollars": risk_dollars,
      "spendableAfterReserve": spendable,
      "atr": atr_val,
      "atrMultiple": atr_multiple,
      "stopDistance": stop_distance,
      "stopPrice": stop_price,
      "targetPrice": target_price,
      "rr": effective_rr,
      "rrBase": target_rr,
      "volRrMultiplier": vol_rr_multiplier,
      "dailyAtrPct": daily_atr_pct,
      "size": size,
      "notionalUsd": notional,
      "notionalUsdWithFee": notional_with_fee,
      "rawNotionalUsd": notional_unclipped,
      "tradesToday": trades_today,
      "maxTradesPerDay": cfg.trading.max_trades_per_symbol_per_day,
      "warnings": warnings,
      "feeRate": fee_rate,
    }


  # ── Futures trading — orders, stops & positions ─────────────────────────────────────────────────
  @function_tool
  async def place_futures_market_order(
    symbol: str,
    side: str,
    notional_usd: float,
    leverage: float = 1.0,
    size_override: float | None = None,
    confidence: float | None = None,
    rationale: str | None = None,
    stop: str | None = None,
    stop_price_type: str | None = None,
    stop_price: float | None = None,
    take_profit_price: float | None = None,
    stop_loss_price: float | None = None,
    reduce_only: bool | None = None,
    auto_protect: bool = True,
  ) -> Dict[str, Any]:
    """Place a linear futures market order using notional and leverage (supports optional attached TP/SL). Falls back to paper if futures disabled."""
    spot_symbol = _resolve_allowed_spot_symbol(symbol, allowed_symbols)
    if not spot_symbol:
      spot_symbol = _repair_allowed_symbol(symbol)
    if not spot_symbol:
      return {"error": "Unsupported symbol", "allowed": sorted(allowed_symbols), "requested": symbol}

    is_entry = not reduce_only
    # Circuit breaker: block new entries when trading is restricted
    if is_entry and snapshot.trading_restricted:
      return {"rejected": True, "reason": f"Trading restricted (close-only mode): {snapshot.restriction_reason}", "hint": "Only reduce_only/close operations are allowed during circuit breaker activation."}

    # Post-loss cooldown: prevent revenge trading on same symbol
    if is_entry and cfg.trading.post_loss_cooldown_minutes > 0:
      last_loss_ts = memory.last_loss_time(spot_symbol)
      if last_loss_ts:
        elapsed_min = (int(time.time()) - last_loss_ts) / 60
        if elapsed_min < cfg.trading.post_loss_cooldown_minutes:
          remaining = int(cfg.trading.post_loss_cooldown_minutes - elapsed_min)
          return {"rejected": True, "reason": f"Post-loss cooldown active for {spot_symbol} ({remaining}min remaining)", "hint": "Prevents revenge trading after a stop-loss. Try a different symbol or wait for cooldown to expire."}

    # No-chase after a win: don't re-enter the same direction at a worse price than a recent profit-take.
    if is_entry and cfg.profit_protection.no_chase_enabled:
      win = memory.recent_win_close(spot_symbol, cfg.profit_protection.post_win_cooldown_minutes)
      ref_ticker = snapshot.tickers.get(spot_symbol)
      ref_px = float(ref_ticker.price) if ref_ticker else 0.0
      if win and win.get("exitPrice") and ref_px > 0 and should_block_chase(
        close_type=win.get("closeType") or "",
        exit_price=float(win["exitPrice"]),
        new_side=side,
        new_price=ref_px,
        buffer_pct=cfg.profit_protection.no_chase_buffer_pct,
      ):
        return {"rejected": True, "reason": f"No-chase: just took profit on {spot_symbol} near {float(win['exitPrice']):.6g}; not re-entering {side} at a worse price ({ref_px:.6g})", "hint": "You recently closed this in profit. Re-enter only on a genuine pullback (better than your exit) or after the post-win cooldown expires."}

    # Post-trade cooldown: minimum interval between trades on same symbol
    if is_entry and cfg.trading.min_trade_interval_minutes > 0:
      last_trade_ts = memory.last_trade_time(spot_symbol)
      if last_trade_ts:
        elapsed_min = (int(time.time()) - last_trade_ts) / 60
        if elapsed_min < cfg.trading.min_trade_interval_minutes:
          remaining = int(cfg.trading.min_trade_interval_minutes - elapsed_min)
          return {"rejected": True, "reason": f"Trade interval cooldown for {spot_symbol} ({remaining}min remaining)", "hint": f"Minimum {cfg.trading.min_trade_interval_minutes:.0f}min between trades on same symbol to prevent overtrading."}

    # Daily gate enforcement: block entries opposing the 1D trend (unless exhausted)
    if is_entry and spot_symbol in _daily_gate_state:
      gate = _daily_gate_state[spot_symbol]
      daily_bias = gate.get("daily_bias", "neutral")
      daily_bias_raw = gate.get("daily_bias_raw", daily_bias)
      daily_exhausted = gate.get("daily_exhausted", False)
      side_lower = (side or "").lower()
      if daily_exhausted and daily_bias_raw in ("bullish", "bearish"):
        is_continuation = (
          (daily_bias_raw == "bullish" and side_lower == "buy") or
          (daily_bias_raw == "bearish" and side_lower == "sell")
        )
        if is_continuation:
          if allow_trend_aligned_short(
            daily_exhausted=daily_exhausted, daily_bias_raw=daily_bias_raw, side=side_lower,
            bias_1h=gate.get("intraday_bias_1h", "neutral"), bias_15m=gate.get("intraday_bias_15m", "neutral"),
            confidence=confidence, cfg=cfg.regime,
          ):
            logger.info("TREND-SHORT ALLOWED: %s %s — exhausted-bearish daily but 1h/15m confirm downtrend resumption (conf=%.2f)", side_lower, spot_symbol, confidence or 0.0)
          else:
            logger.warning("ANTI-FOMO BLOCK: %s %s rejected — daily %s exhausted (RSI extreme)", side, spot_symbol, daily_bias_raw)
            return {"rejected": True, "reason": f"Daily exhaustion: {daily_bias_raw} trend overextended — no continuation entry", "hint": "Daily RSI is at an extreme. Wait for the pullback or trade counter-trend with a clear reversal signal."}
      if daily_bias != "neutral" and not daily_exhausted:
        opposing = (daily_bias == "bearish" and side_lower == "buy") or (daily_bias == "bullish" and side_lower == "sell")
        if opposing:
          if allow_reversal_long(
            daily_bias=daily_bias, side=side_lower,
            bias_1h=gate.get("intraday_bias_1h", "neutral"), bias_15m=gate.get("intraday_bias_15m", "neutral"),
            confidence=confidence, cfg=cfg.regime,
          ):
            logger.info("REVERSAL LONG ALLOWED: %s %s — bearish daily but 1h/15m confirm a turn (conf=%.2f)", side, spot_symbol, confidence or 0.0)
          else:
            logger.warning("DAILY GATE BLOCK: %s %s rejected — 1D trend is %s", side, spot_symbol, daily_bias)
            return {"rejected": True, "reason": f"Daily gate: 1D trend is {daily_bias} — {side} entry blocked", "daily_bias": daily_bias, "hint": "The 1D timeframe opposes this trade direction. Only trade WITH the daily trend, take a confirmed reversal (1h+15m turned, high confidence), or wait for it to turn neutral."}
      # 1h alignment: block entries when 1h bias opposes the proposed side (catches bounces in confirmed corrections)
      intraday_bias_1h = gate.get("intraday_bias_1h", "neutral")
      intraday_1h_opposes = (
        (intraday_bias_1h == "bearish" and side_lower == "buy") or
        (intraday_bias_1h == "bullish" and side_lower == "sell")
      )
      if intraday_1h_opposes:
        if resolve_gate_deadlock(
          daily_bias=daily_bias, daily_exhausted=daily_exhausted, side=side_lower,
          bias_1h=intraday_bias_1h, bias_15m=gate.get("intraday_bias_15m", "neutral"),
          confidence=confidence, cfg=cfg.regime,
        ):
          logger.info("DEADLOCK BREAK: %s %s — daily-aligned entry allowed past stalling 1h counter-bounce (daily=%s, 1h=%s, conf=%.2f)", side, spot_symbol, daily_bias, intraday_bias_1h, confidence or 0.0)
        else:
          logger.warning("1H ALIGN BLOCK: %s %s rejected — 1h bias %s opposes %s", side, spot_symbol, intraday_bias_1h, side_lower)
          return {"rejected": True, "reason": f"1h trend is {intraday_bias_1h} — {side_lower} entry blocked", "hint": "1h timeframe opposes this direction. The daily trend is in correction, not a healthy pullback. Do NOT enter against the 1h trajectory even when daily aligns."}
      # Timeframe-conflict gate: catches 15m vs higher-TF disagreement not already blocked by 1h alignment
      tf_conflict = gate.get("timeframe_conflict", False)
      intraday_bias_15m = gate.get("intraday_bias_15m", "neutral")
      if tf_conflict and intraday_bias_15m != "neutral":
        intraday_opposes = (
          (intraday_bias_15m == "bearish" and side_lower == "buy") or
          (intraday_bias_15m == "bullish" and side_lower == "sell")
        )
        if intraday_opposes:
          logger.warning("TF CONFLICT BLOCK: %s %s rejected — daily/intraday split, 15m %s opposes %s", side, spot_symbol, intraday_bias_15m, side_lower)
          return {"rejected": True, "reason": f"Timeframe conflict: 15m {intraday_bias_15m} opposes proposed {side_lower}", "hint": "Wait for 15m to align with the higher-TF bias, or pick a different symbol."}

    # Correlation gate: block alt LONGs while BTC's daily regime is bearish (alts are high-beta to
    # BTC; longing them into a BTC downtrend is the RE-USDT failure mode).
    if is_entry and block_alt_long_in_btc_downtrend(symbol=spot_symbol, side=side, btc_daily_bias=_btc_daily_bias(), cfg=cfg.regime):
      logger.warning("CORRELATION GATE BLOCK: long %s rejected — BTC daily regime is bearish", spot_symbol)
      return {"rejected": True, "reason": f"Correlation gate: BTC daily regime is bearish — long on alt {spot_symbol} blocked", "hint": "Alts are high-beta to BTC; do NOT long an altcoin while BTC's daily trend is down. Trade a trend-aligned short, wait for BTC's daily to turn, or trade a major instead."}

    # Symbol bench: a symbol that keeps losing is quarantined until its cooldown lifts —
    # stops the re-take-the-same-losing-trade loop (4 straight ETH short stop-outs, Jul 1-2).
    if is_entry:
      _bench_until = _edge_state()["bench"].get(spot_symbol, 0)
      if _bench_until > int(time.time()):
        _bench_hours = (_bench_until - int(time.time())) / 3600.0
        logger.warning("SYMBOL BENCH BLOCK: %s %s rejected — repeated recent losses, benched %.1fh more", side, spot_symbol, _bench_hours)
        return {"rejected": True, "reason": f"Symbol benched: {spot_symbol} lost repeatedly in its recent closes — no entries for {_bench_hours:.1f}h more", "symbol": spot_symbol, "hint": "This symbol keeps stopping out; the setup that looks compliant is not working in the current tape. Trade a different symbol or stand aside — the bench lifts automatically."}

    # Volatility filter: soft-scale position below 1.5× threshold; hard-block above
    _atr_scale_fm = 1.0
    if is_entry and spot_symbol in _daily_gate_state:
      gate = _daily_gate_state[spot_symbol]
      daily_atr_pct = gate.get("daily_atr_pct")
      if daily_atr_pct is not None and daily_atr_pct > cfg.trading.max_atr_pct_for_entry:
        hard_limit = cfg.trading.max_atr_pct_for_entry * 1.5
        if daily_atr_pct > hard_limit:
          logger.warning("VOLATILITY BLOCK: %s rejected — ATR=%.2f%% exceeds hard limit %.2f%%", spot_symbol, daily_atr_pct, hard_limit)
          return {"rejected": True, "reason": f"Extreme volatility: ATR={daily_atr_pct:.2f}% > {hard_limit:.1f}% hard limit", "hint": "Wait for volatility to settle before entering."}
        _atr_scale_fm = max(0.30, (cfg.trading.max_atr_pct_for_entry / daily_atr_pct) ** 2)
        logger.info("VOLATILITY SOFT GATE: futures %s ATR=%.2f%% — scaling position to %.0f%%", spot_symbol, daily_atr_pct, _atr_scale_fm * 100)

    # Regime throttle (B): shrink size + raise the confidence bar in a hostile (bearish/exhausted) daily.
    _g_fm = _daily_gate_state.get(spot_symbol, {})
    if is_entry:
      _atr_scale_fm = max(0.30, _atr_scale_fm * regime_size_factor(_g_fm.get("daily_bias", "neutral"), _g_fm.get("daily_exhausted", False), cfg.regime))

    # Confidence enforcement: reject entries below the (regime-adjusted) minimum confidence
    if is_entry and confidence is not None:
      _eff_min_fm = effective_min_confidence(cfg.trading.min_confidence, _g_fm.get("daily_bias", "neutral"), _g_fm.get("daily_exhausted", False), cfg.regime)
      if confidence < _eff_min_fm:
        return {"rejected": True, "reason": f"Confidence {confidence:.2f} below regime-adjusted minimum {_eff_min_fm:.2f}", "hint": "Only enter with sufficient conviction; the bar is raised in a bearish/exhausted regime. Analyze another coin or wait for a better setup."}
      # Conviction sizing: shrink low-conviction entries (confidence barely above the floor), ramping
      # to full size as conviction rises — targets the full-size low-conviction shorts that drained SOL.
      _conv_scale_fm = conviction_size_factor(confidence, _eff_min_fm, cfg.regime)
      if _conv_scale_fm < 1.0:
        logger.info("CONVICTION SIZING: futures %s conf=%.2f (floor %.2f) — scaling position to %.0f%%", spot_symbol, confidence, _eff_min_fm, _conv_scale_fm * 100)
        _atr_scale_fm = max(0.30, _atr_scale_fm * _conv_scale_fm)

    # Loss-streak throttle: consecutive realized losses shrink the next entry (anti-martingale).
    if is_entry:
      _streak_fm = _edge_state()["size_factor"]
      if _streak_fm < 1.0:
        logger.info("LOSS-STREAK THROTTLE: futures %s — sizing to %.0f%% after %d consecutive losses", spot_symbol, _streak_fm * 100, _edge_state()["stats"].get("loss_streak", 0))
        _atr_scale_fm = max(0.30, _atr_scale_fm * _streak_fm)

    # Anti-stacking: block add-on entries when existing position is losing
    if is_entry and snapshot.futures_positions:
      futures_symbol_check = _to_futures_symbol(spot_symbol)
      if futures_symbol_check:
        existing = next((p for p in snapshot.futures_positions if p.get("symbol") == futures_symbol_check), None)
        if existing:
          unrealized = _to_float(existing.get("unrealisedPnl")) or _to_float(existing.get("unrealizedPnl")) or 0
          pos_qty = _to_float(existing.get("currentQty")) or 0
          same_direction = (pos_qty > 0 and (side or "").lower() == "buy") or (pos_qty < 0 and (side or "").lower() == "sell")
          if same_direction and unrealized < 0:
            logger.warning("ANTI-STACKING: %s %s add-on rejected — existing position unrealizedPnl=%.4f", side, spot_symbol, unrealized)
            return {"rejected": True, "reason": f"Cannot add to losing {spot_symbol} position (unrealizedPnl={unrealized:.4f})", "hint": "Do NOT add to losing positions. Wait for the position to recover or close it first, then trade a different symbol."}
          # Anti-FOMO stacking: block adds when daily is exhausted in same direction (any PnL)
          if same_direction and spot_symbol in _daily_gate_state:
            stack_gate = _daily_gate_state[spot_symbol]
            if stack_gate.get("daily_exhausted", False):
              stack_bias_raw = stack_gate.get("daily_bias_raw", "neutral")
              chasing_top = (stack_bias_raw == "bullish" and pos_qty > 0) or (stack_bias_raw == "bearish" and pos_qty < 0)
              if chasing_top:
                logger.warning("ANTI-FOMO STACKING: %s %s add-on rejected — daily %s exhausted", side, spot_symbol, stack_bias_raw)
                return {"rejected": True, "reason": f"No adds when daily {stack_bias_raw} is exhausted (RSI extreme)", "hint": "Protect the existing position with TP/SL; do not chase the top/bottom."}

    if not cfg.kucoin_futures.enabled or not kucoin_futures:
      return {"paper": True, "reason": "Futures disabled in config"}
    try:
      notional_input = float(notional_usd or 0) * _atr_scale_fm
      lev_requested = float(leverage or 0)
    except (TypeError, ValueError):
      return {"error": "Invalid notional or leverage"}
    if notional_input <= 0 or lev_requested <= 0:
      return {"error": "Invalid notional or leverage"}
    # Concentration cap: shrink a single position's notional to <= max_position_equity_pct of total
    # equity regardless of leverage — bounds blast radius (RE-USDT was ~74% of equity in one name).
    if is_entry:
      _conc_scale = concentration_scale(notional_input, snapshot.total_usdt, cfg.trading.max_position_equity_pct)
      if _conc_scale < 1.0:
        logger.info("CONCENTRATION CAP: futures %s notional %.2f → %.2f (<= %.0f%% of equity %.2f)",
                    spot_symbol, notional_input, notional_input * _conc_scale, cfg.trading.max_position_equity_pct * 100, snapshot.total_usdt)
        notional_input *= _conc_scale
    rationale_norm = (rationale or "").lower() if rationale else ""
    trades_today = memory.trades_today(spot_symbol)
    if trades_today >= cfg.trading.max_trades_per_symbol_per_day:
      return {
        "rejected": True,
        "reason": "Daily trade cap reached",
        "tradesToday": trades_today,
        "limit": cfg.trading.max_trades_per_symbol_per_day,
      }
    if cfg.trading.sentiment_filter_enabled:
      latest_sent = memory.latest_sentiment(symbol)
      day_key = int(time.time() // 86400)
      if not latest_sent or latest_sent.get("day") != day_key:
        return {"rejected": True, "reason": "Sentiment missing for today", "minScore": cfg.trading.sentiment_min_score}
      if latest_sent.get("score", 0) < cfg.trading.sentiment_min_score:
        return {
          "rejected": True,
          "reason": "Sentiment below threshold",
          "score": latest_sent.get("score"),
          "minScore": cfg.trading.sentiment_min_score,
        }

    futures_symbol = _to_futures_symbol(spot_symbol)
    if not futures_symbol:
      return {"error": "Invalid symbol for futures", "symbol": spot_symbol}

    contract = _get_contract_spec(futures_symbol)
    if not contract:
      return {"error": "Futures contract spec unavailable", "futuresSymbol": futures_symbol}

    # New-listing guard: skip freshly-listed, thin, ultra-volatile contracts (the RE-USDT trap).
    if is_entry and cfg.trading.min_futures_listing_age_days > 0:
      _first_open = _to_float(contract.get("firstOpenDate"))
      _first_open_s = _first_open / 1000.0 if _first_open > 1e12 else _first_open
      if _first_open_s > 0:
        _age_days = (time.time() - _first_open_s) / 86400.0
        if _age_days < cfg.trading.min_futures_listing_age_days:
          logger.warning("NEW-LISTING BLOCK: %s rejected — contract age %.1fd < %.1fd minimum", futures_symbol, _age_days, cfg.trading.min_futures_listing_age_days)
          return {"rejected": True, "reason": f"New-listing guard: {futures_symbol} is {_age_days:.1f}d old (< {cfg.trading.min_futures_listing_age_days:.0f}d minimum)", "hint": "Freshly-listed perps are thin and ultra-volatile (this is how RE-USDT blew up). Wait for a real price history, or trade an established contract."}

    price = float(snapshot.tickers[spot_symbol].price)
    if price <= 0:
      return {"error": "Invalid or missing price"}

    multiplier = _to_float(contract.get("multiplier"))
    lot_size = int(contract.get("lotSize") or 1)
    max_order_qty = contract.get("maxOrderQty")
    contract_max_leverage = _to_float(contract.get("maxLeverage")) or None
    fee_rate = _to_float(contract.get("takerFeeRate")) or fees.get("futures_taker", 0.0006)
    slippage_rate = cfg.trading.estimated_slippage_pct
    if multiplier <= 0:
      return {"error": "Invalid contract multiplier", "contract": contract}
    leverage_caps = [
      c for c in (
        snapshot.max_leverage,
        cfg.trading.max_leverage,
        contract_max_leverage,
      )
      if c and c > 0
    ]
    if is_entry and cfg.trading.max_entry_leverage > 0:
      leverage_caps.append(cfg.trading.max_entry_leverage)
    max_leverage_cap = min(leverage_caps) if leverage_caps else 3.0
    lev = min(max(1.0, lev_requested), max_leverage_cap)
    leverage_clamped = lev < lev_requested - 1e-9
    if leverage_clamped and is_entry:
      logger.info("Entry leverage capped: requested %.1fx → applied %.1fx (max_entry_leverage=%.1f)", lev_requested, lev, cfg.trading.max_entry_leverage)

    try:
      base_size = float(size_override) if size_override is not None else notional_input / price
    except (TypeError, ValueError):
      return {"error": "Invalid size_override"}
    if base_size <= 0:
      return {"error": "Computed size is zero"}
    
    # Check if this is a closing/reducing trade and validate profit
    closing_pnl = None
    closing_roi = None
    is_closing_trade = False
    existing_pos = next((p for p in snapshot.futures_positions if p.get("symbol") == futures_symbol), None)
    if existing_pos:
      try:
        pos_qty = float(existing_pos.get("currentQty") or 0)
        pos_entry = float(existing_pos.get("avgEntryPrice") or 0)
        
        # Determine if we are closing (opposite side or reduceOnly)
        is_long = pos_qty > 0
        is_short = pos_qty < 0
        is_closing = False
        
        if reduce_only:
          is_closing = True
        elif is_long and side == "sell":
          is_closing = True
        elif is_short and side == "buy":
          is_closing = True
          
        is_closing_trade = is_closing
        if is_closing and pos_entry > 0:
          # Calculate estimated PnL on the portion being closed
          # Note: base_size is the amount we are TENTATIVELY sending
          # We should clip it to position size to be accurate about what's being closed
          close_size = min(abs(pos_qty), base_size)
          
          if close_size > 0:
            # Entry value = size * entry
            # Exit value = size * price
            # Long PnL = (Exit - Entry)
            # Short PnL = (Entry - Exit)
            
            if is_long:
              raw_pnl = (price - pos_entry) * close_size
            else:
              raw_pnl = (pos_entry - price) * close_size
              
            # Deduct fees on both entry and exit for accurate net PnL.
            entry_fee = pos_entry * close_size * (fee_rate + slippage_rate)
            exit_fee = price * close_size * (fee_rate + slippage_rate)
            net_pnl = raw_pnl - entry_fee - exit_fee
            
            entry_value = pos_entry * close_size
            roi_pct = 0.0
            if entry_value > 0:
              roi_pct = net_pnl / entry_value
            
            closing_pnl = net_pnl
            closing_roi = roi_pct
              
            # Re-check allow_loss rationale
            rationale_norm = (rationale or "").lower() if rationale else ""
            allow_loss_keyword = any(term in rationale_norm for term in ("stop loss", "cut loss", "emergency", "liquidate", "portfolio review", "close position", "rebalance", "force sell", "force-sell", "supervisor"))
            
            min_profit_usd = cfg.trading.min_net_profit_usd
            min_roi = cfg.trading.min_profit_roi_pct
            
            # For closing trades: allow any non-negative PnL. Only block actual losses without explicit keyword.
            if net_pnl < 0 and not allow_loss_keyword:
               return {
                "rejected": True,
                "reason": f"Closing at a loss (expected PnL ${net_pnl:.4f}); include 'stop loss' or 'cut loss' in rationale to allow",
                "expectedPnl": net_pnl,
                "expectedRoi": roi_pct,
                "minProfitUsd": min_profit_usd,
                "minRoiPct": min_roi,
                "entryPrice": pos_entry,
                "currentPrice": price,
                "closeSize": close_size,
                "hint": "Retry with 'cut loss' or 'stop loss' in rationale to allow closing at a loss.",
               }
            logger.info("Closing Futures %s - Expected PnL: %.4f USD (ROI: %.2f%%)", futures_symbol, net_pnl, roi_pct * 100)

      except Exception as exc:
        logger.warning("Failed to calc closing PnL for %s: %s", futures_symbol, exc)

    contracts_raw = base_size / multiplier
    lot = max(1, lot_size)
    contracts = int(math.ceil(contracts_raw / lot) * lot)
    actual_notional = contracts * multiplier * price
    min_notional = lot * multiplier * price
    if actual_notional < min_notional:
      return {
        "rejected": True,
        "reason": "Below contract minimum notional",
        "minNotionalUsd": min_notional,
        "requestedNotionalUsd": notional_input,
      }
    if max_order_qty and isinstance(max_order_qty, (int, float)) and contracts > max_order_qty:
      return {"rejected": True, "reason": "Exceeds max order size", "maxContracts": max_order_qty, "contracts": contracts}
    notional = actual_notional
    if notional > snapshot.max_position_usd * max_leverage_cap:
      return {
        "rejected": True,
        "reason": "Exceeds max notional cap",
        "cap": snapshot.max_position_usd * max_leverage_cap,
        "notional": notional,
        "maxLeverage": max_leverage_cap,
      }
    # Prefer live contract fee if available.
    notional_with_fee = notional * (1 + fee_rate + slippage_rate)
    def _as_price(val: Any) -> float | None:
      try:
        p = float(val)
        return p if p > 0 else None
      except Exception:
        return None

    tp_val = _as_price(take_profit_price)
    sl_val = _as_price(stop_loss_price)
    trigger_stop_val = _as_price(stop_price)
    min_edge_pct = max(0.003, 5 * (fee_rate + slippage_rate))  # at least 0.3% or 5x friction cost
    if not reduce_only:
      edge_candidates = [
        abs(tp_val - price) / price if tp_val else None,
        abs(sl_val - price) / price if sl_val else None,
        abs(trigger_stop_val - price) / price if trigger_stop_val else None,
      ]
      edge_candidates = [c for c in edge_candidates if c is not None]
      if edge_candidates:
        min_edge_found = min(edge_candidates)
        if min_edge_found < min_edge_pct:
          min_tp = price * (1 + min_edge_pct) if side == "buy" else price * (1 - min_edge_pct)
          min_sl = price * (1 - min_edge_pct) if side == "buy" else price * (1 + min_edge_pct)
          return {
            "rejected": True,
            "reason": "TP/SL distance too tight relative to fees",
            "minEdgePct": min_edge_pct,
            "edgePct": min_edge_found,
            "price": price,
            "takeProfit": tp_val,
            "stopLoss": sl_val or trigger_stop_val,
            "feeRate": fee_rate,
            "hint": f"Widen TP to at least {min_tp:.2f} and SL to at least {min_sl:.2f} (>{min_edge_pct*100:.2f}% from entry), then retry.",
          }
      # Reward:risk floor — don't take entries that risk more than they aim to make. Futures had no
      # RR check (only spot did), which is why losses ran ~4x the wins. Enforced when TP+SL are both set.
      _rr_stop_fm = sl_val or trigger_stop_val
      if cfg.trading.min_futures_rr > 0 and tp_val and _rr_stop_fm:
        _req_rr_fm = _edge_state()["required_rr"]
        _rr_fm = reward_risk_ratio(side, price, tp_val, _rr_stop_fm)
        if _rr_fm is None:
          logger.warning("RR BLOCK: futures %s %s rejected — TP/SL on the wrong side of entry", side, spot_symbol)
          return {"rejected": True, "reason": "TP/SL on the wrong side of entry — cannot form a valid bracket", "price": price, "takeProfit": tp_val, "stopLoss": _rr_stop_fm, "hint": "For a long, TP must be above and SL below entry; reverse for a short."}
        if _rr_fm < _req_rr_fm:
          logger.warning("RR BLOCK: futures %s %s rejected — reward:risk %.2f < %.2f", side, spot_symbol, _rr_fm, _req_rr_fm)
          return {"rejected": True, "reason": f"Reward:risk {_rr_fm:.2f} below minimum {_req_rr_fm:.2f}", "rr": _rr_fm, "minRr": _req_rr_fm, "price": price, "takeProfit": tp_val, "stopLoss": _rr_stop_fm, "hint": f"Widen the take-profit or tighten the stop so the TP distance is at least {_req_rr_fm:.1f}x the stop distance (floor rises while recent trades are net-losing). Skip setups that can't clear it."}
    base_size = contracts * multiplier
    expected_tp_pnl = None
    if tp_val and base_size > 0:
      if side == "buy":
        expected_tp_pnl = (tp_val * (1 - fee_rate - slippage_rate) - price * (1 + fee_rate + slippage_rate)) * base_size
      else:
        expected_tp_pnl = (price * (1 - fee_rate - slippage_rate) - tp_val * (1 + fee_rate + slippage_rate)) * base_size
    
    rationale_norm = (rationale or "").lower() if rationale else ""
    allow_loss = reduce_only or any(term in rationale_norm for term in ("stop loss", "cut loss", "emergency", "liquidate", "portfolio review", "close position", "rebalance", "force sell", "force-sell", "supervisor"))
    
    min_profit_usd = cfg.trading.min_net_profit_usd
    min_roi = cfg.trading.min_profit_roi_pct
    futures_entry_cost = price * base_size * (1 + fee_rate + slippage_rate)
    expected_tp_roi = expected_tp_pnl / futures_entry_cost if expected_tp_pnl is not None and futures_entry_cost > 0 else None

    if expected_tp_pnl is not None and expected_tp_pnl < min_profit_usd and (expected_tp_roi is None or expected_tp_roi < min_roi) and not allow_loss:
      return {
        "rejected": True,
        "reason": f"Take-profit too close; requires at least ${min_profit_usd:.2f} net profit or {min_roi*100:.2f}% ROI (after slippage)",
        "expectedTpPnl": expected_tp_pnl,
        "expectedTpRoi": expected_tp_roi,
        "minProfitUsd": min_profit_usd,
        "minRoiPct": min_roi,
        "size": base_size,
        "price": price,
        "tp": tp_val,
        "feeRate": fee_rate,
        "slippageRate": slippage_rate,
        "hint": f"Widen TP farther from entry (min {min_edge_pct*100:.2f}% away), increase notional, or increase leverage to amplify the edge.",
      }

    protection_stop_val = sl_val or trigger_stop_val
    protection_tp_val = tp_val
    protection_price_type = (stop_price_type or "MP").upper() if (stop_price_type or "").upper() in {"TP", "MP", "IP"} else "MP"
    # Ensure required margin is funded; auto-transfer from spot if futures are empty.
    spot_accounts = kucoin.get_accounts()
    spot_balance_map: Dict[str, float] = {}
    for bal in spot_accounts:
      spot_balance_map[bal.currency] = spot_balance_map.get(bal.currency, 0.0) + float(bal.available or 0)

    futures_overview: Dict[str, Any] | None = None
    futures_available = 0.0
    margin_mode: str | None = None
    try:
      position_info = kucoin_futures.get_position(futures_symbol)
      if isinstance(position_info, dict) and "crossMode" in position_info:
        margin_mode = "cross" if position_info.get("crossMode") else "isolated"
    except Exception as exc:
      logger.warning("Futures position lookup failed; margin mode unknown: %s", exc)
    # Honor the configured margin mode (default cross); "auto" preserves the existing position's mode.
    if _futures_margin_mode in ("cross", "isolated"):
      margin_mode = _futures_margin_mode
    elif margin_mode is None:
      margin_mode = "cross"
    if kucoin_futures:
      try:
        futures_overview = kucoin_futures.get_account_overview()
        futures_available = _to_float(futures_overview.get("availableBalance"))
      except Exception as exc:
        logger.warning("Futures overview unavailable: %s", exc)

    margin_needed = 0.0 if reduce_only else notional / lev
    fee_allowance = notional * (fee_rate + slippage_rate)
    buffer = max(1.0, margin_needed * 0.01)
    required_futures_balance = margin_needed + fee_allowance + buffer

    transfer_used: dict[str, Any] | None = None
    if not snapshot.paper_trading and futures_available < required_futures_balance:
      need = required_futures_balance - futures_available
      
      # Try pulling from Financial (Earn) account first
      financial_available = 0.0
      if snapshot.financial_accounts:
        financial_totals = _aggregate_account_totals(snapshot.financial_accounts)
        financial_available = financial_totals.get("USDT", {}).get("available", 0.0)
      
      if financial_available > 0:
        financial_reserve = financial_available * 0.10
        financial_spendable = max(0.0, financial_available - financial_reserve)
        transfer_amt = min(need, financial_spendable)
        if transfer_amt > 0:
          try:
            transfer_used = kucoin.transfer_funds(
              currency="USDT",
              amount=transfer_amt,
              from_account="financial",
              to_account="contract",
            )
            futures_available += transfer_amt
            need = max(0, required_futures_balance - futures_available)
          except Exception as exc:
            logger.warning("Financial->futures transfer failed: %s", exc)

      # If still need more, try pulling from Spot (Trade) account
      if need > 0:
        spot_usdt = spot_balance_map.get("USDT", 0.0)
        spot_reserve = spot_usdt * 0.10
        spot_spendable = max(0.0, spot_usdt - spot_reserve)
        transfer_amt = min(spot_spendable, need)
        if transfer_amt > 0:
          try:
            transfer_used = kucoin.transfer_funds(
              currency="USDT",
              amount=transfer_amt,
              from_account="trade",
              to_account="contract",
            )
            futures_available += transfer_amt
          except Exception as exc:
            logger.warning("Spot->futures transfer failed: %s", exc)

      # Re-check actual futures balance after transfers.
      if transfer_used and kucoin_futures:
        try:
          post_overview = kucoin_futures.get_account_overview()
          futures_available = _to_float(post_overview.get("availableBalance"))
        except Exception as exc:
          logger.warning("Post-transfer futures balance check failed: %s", exc)
    if required_futures_balance > futures_available and not snapshot.paper_trading:
      spot_usdt_available = spot_balance_map.get("USDT", 0.0)
      shortfall = required_futures_balance - futures_available
      return {
        "rejected": True,
        "reason": "Insufficient futures margin after transfer attempt",
        "requiredBalance": required_futures_balance,
        "available": futures_available,
        "spotAvailable": spot_usdt_available,
        "transferAttempt": transfer_used,
        "hint": f"Need {shortfall:.2f} more USDT in futures. Try: transfer_funds('spot_to_futures', amount={min(shortfall, spot_usdt_available):.2f}), or sell a spot position to free capital, or reduce notional/increase leverage.",
      }

    def _build_order(mode: str | None) -> KucoinFuturesOrderRequest:
      # KuCoin docs show uppercase margin modes; include autoDeposit for isolated.
      mode_norm = None
      auto_deposit = None
      if mode:
        mode_norm = mode.upper()
        if mode_norm == "ISOLATED":
          auto_deposit = False
      return KucoinFuturesOrderRequest(
        symbol=futures_symbol,
        side="buy" if side == "buy" else "sell",
        type="market",
        leverage=str(int(lev)),
        size=str(contracts),  # Kucoin futures expects integer contract size (contracts)
        clientOid=str(uuid.uuid4()),
        marginMode=mode_norm,
        autoDeposit=auto_deposit,
        reduceOnly=reduce_only,
      )

    async def _create_futures_bracket() -> Dict[str, Any]:
      if reduce_only or is_closing_trade or not auto_protect:
        return {}
      bracket: Dict[str, Any] = {}
      exit_side = "sell" if side == "buy" else "buy"
      size_for_exit = contracts * multiplier
      if protection_stop_val:
        bracket["stopLoss"] = await _place_futures_stop_order_impl(
          symbol=spot_symbol,
          side=exit_side,
          leverage=lev,
          size=size_for_exit,
          stop_price=protection_stop_val,
          stop="down" if side == "buy" else "up",
          stop_price_type=protection_price_type,
          reduce_only=True,
          close_order=True,
          order_type="market",
          client_oid=f"{futures_symbol.lower()}-sl-{uuid.uuid4().hex[:12]}",
        )
      if protection_tp_val:
        bracket["takeProfit"] = await _place_futures_stop_order_impl(
          symbol=spot_symbol,
          side=exit_side,
          leverage=lev,
          size=size_for_exit,
          stop_price=protection_tp_val,
          stop="up" if side == "buy" else "down",
          stop_price_type=protection_price_type,
          reduce_only=True,
          close_order=True,
          order_type="market",
          client_oid=f"{futures_symbol.lower()}-tp-{uuid.uuid4().hex[:12]}",
        )
      return bracket

    if snapshot.paper_trading:
      order_req = _build_order(margin_mode)
      record = memory.record_trade(symbol, side, notional, paper=True, price=price, size=contracts * multiplier, venue="futures")
      decision = None
      if confidence is not None:
        decision = memory.log_decision(
          symbol,
        f"futures_{side}",
        float(confidence),
        rationale or "paper trade",
        pnl=None,
        paper=True,
      )
      return_val = {
        "paper": True,
        "orderRequest": order_req.__dict__,
        "tradeRecord": record,
        "decisionLog": decision,
        "rationale": rationale,
        "feeRate": fee_rate,
        "notionalWithFee": notional_with_fee,
        "futuresSymbol": futures_symbol,
        "contracts": contracts,
        "contractSpec": {"multiplier": multiplier, "lotSize": lot_size},
        "appliedLeverage": lev,
        "leverageClampedFrom": lev_requested if leverage_clamped else None,
        "maxLeverageCap": max_leverage_cap,
      }
      bracket = await _create_futures_bracket()
      if bracket:
        return_val["bracket"] = bracket
      if closing_pnl is not None:
        return_val["closingPnl"] = closing_pnl
        return_val["closingRoi"] = closing_roi
      return return_val

    attempts: list[Dict[str, Any]] = []
    order_req = _build_order(margin_mode)
    try:
      kucoin_futures.set_margin_mode(futures_symbol, (margin_mode or "cross"))
    except Exception as exc:
      logger.warning("set_margin_mode failed (continuing): %s", exc)
    _apply_cross_leverage(futures_symbol, lev, margin_mode)
    try:
      res = kucoin_futures.place_order(order_req).__dict__
      record = memory.record_trade(spot_symbol, side, notional, paper=False, price=price, size=contracts * multiplier, venue="futures")
      res["tradeRecord"] = record
      if confidence is not None:
        res["decisionLog"] = memory.log_decision(
          spot_symbol,
          f"futures_{side}",
          float(confidence),
          rationale or "live trade",
          pnl=None,
          paper=False,
        )
      res["rationale"] = rationale
      res["feeRate"] = fee_rate
      res["notionalWithFee"] = notional_with_fee
      res["futuresSymbol"] = futures_symbol
      res["contracts"] = contracts
      res["contractSpec"] = {"multiplier": multiplier, "lotSize": lot_size}
      res["transferUsed"] = transfer_used
      res["marginModeUsed"] = margin_mode
      res["appliedLeverage"] = lev
      res["previousAttempts"] = attempts
      res["maxLeverageCap"] = max_leverage_cap
      bracket = await _create_futures_bracket()
      if bracket:
        res["bracket"] = bracket
      if leverage_clamped:
        res["leverageClampedFrom"] = lev_requested
      if closing_pnl is not None:
        res["closingPnl"] = closing_pnl
        res["closingRoi"] = closing_roi
      return res
    except Exception as exc:
      attempts.append({"marginMode": margin_mode, "error": str(exc)})

    # If we reach here, try the opposite mode as a fallback.
    opposite_mode = "isolated" if margin_mode == "cross" else "cross"
    order_req = _build_order(opposite_mode)
    try:
      kucoin_futures.set_margin_mode(futures_symbol, opposite_mode)
    except Exception as exc:
      logger.warning("set_margin_mode fallback failed (continuing): %s", exc)
    _apply_cross_leverage(futures_symbol, lev, opposite_mode, context="fallback")
    try:
      res = kucoin_futures.place_order(order_req).__dict__
      record = memory.record_trade(spot_symbol, side, notional, paper=False, price=price, size=contracts * multiplier, venue="futures")
      res["tradeRecord"] = record
      if confidence is not None:
        res["decisionLog"] = memory.log_decision(
          spot_symbol,
          f"futures_{side}",
          float(confidence),
          rationale or "live trade",
          pnl=None,
          paper=False,
        )
      res["rationale"] = rationale
      res["feeRate"] = fee_rate
      res["notionalWithFee"] = notional_with_fee
      res["futuresSymbol"] = futures_symbol
      res["contracts"] = contracts
      res["contractSpec"] = {"multiplier": multiplier, "lotSize": lot_size}
      res["transferUsed"] = transfer_used
      res["marginModeUsed"] = opposite_mode
      res["appliedLeverage"] = lev
      res["previousAttempts"] = attempts
      res["maxLeverageCap"] = max_leverage_cap
      bracket = await _create_futures_bracket()
      if bracket:
        res["bracket"] = bracket
      if leverage_clamped:
        res["leverageClampedFrom"] = lev_requested
      if closing_pnl is not None:
        res["closingPnl"] = closing_pnl
        res["closingRoi"] = closing_roi
      return res
    except Exception as exc:
      attempts.append({"marginMode": opposite_mode, "error": str(exc)})

    return {
      "error": "Futures order failed",
      "attempts": attempts,
      "futuresSymbol": futures_symbol,
      "contracts": contracts,
      "marginModeDetected": margin_mode,
      "transferUsed": transfer_used,
    }

  @function_tool
  async def place_futures_limit_order(
    symbol: str,
    side: str,
    notional_usd: float,
    entry_price: float,
    leverage: float = 1.0,
    size_override: float | None = None,
    confidence: float | None = None,
    rationale: str | None = None,
    take_profit_price: float | None = None,
    stop_loss_price: float | None = None,
  ) -> Dict[str, Any]:
    """Place a futures limit entry order at a technically derived target price.
    Use for new entries where you want to wait for price to reach a key level
    (EMA resistance/support, Bollinger Band, swing high/low, VWAP) before entering.
    Rejects if entry_price is too close to current price — use place_futures_market_order
    for closes or when price is already at the target level.
    Bracket TP/SL are placed on the NEXT run after the limit fills."""
    spot_symbol = _resolve_allowed_spot_symbol(symbol, allowed_symbols)
    if not spot_symbol:
      spot_symbol = _repair_allowed_symbol(symbol)
    if not spot_symbol:
      return {"error": "Unsupported symbol", "allowed": sorted(allowed_symbols), "requested": symbol}

    try:
      entry_price_val = float(entry_price)
    except (TypeError, ValueError):
      return {"error": "entry_price must be a valid number"}
    if entry_price_val <= 0:
      return {"error": "entry_price must be positive"}

    side_lower = (side or "").lower()
    lev_requested = max(1.0, float(leverage or 1.0))

    if snapshot.trading_restricted:
      return {"rejected": True, "reason": f"Trading restricted (close-only mode): {snapshot.restriction_reason}", "hint": "Only reduce_only/close operations are allowed during circuit breaker activation."}

    if cfg.trading.post_loss_cooldown_minutes > 0:
      last_loss_ts = memory.last_loss_time(spot_symbol)
      if last_loss_ts:
        elapsed_min = (int(time.time()) - last_loss_ts) / 60
        if elapsed_min < cfg.trading.post_loss_cooldown_minutes:
          remaining = int(cfg.trading.post_loss_cooldown_minutes - elapsed_min)
          return {"rejected": True, "reason": f"Post-loss cooldown active for {spot_symbol} ({remaining}min remaining)", "hint": "Try a different symbol or wait."}

    if cfg.trading.min_trade_interval_minutes > 0:
      last_trade_ts = memory.last_trade_time(spot_symbol)
      if last_trade_ts:
        elapsed_min = (int(time.time()) - last_trade_ts) / 60
        if elapsed_min < cfg.trading.min_trade_interval_minutes:
          remaining = int(cfg.trading.min_trade_interval_minutes - elapsed_min)
          return {"rejected": True, "reason": f"Trade interval cooldown for {spot_symbol} ({remaining}min remaining)"}

    if spot_symbol in _daily_gate_state:
      gate = _daily_gate_state[spot_symbol]
      daily_bias = gate.get("daily_bias", "neutral")
      daily_bias_raw = gate.get("daily_bias_raw", daily_bias)
      daily_exhausted = gate.get("daily_exhausted", False)
      if daily_exhausted and daily_bias_raw in ("bullish", "bearish"):
        is_continuation = (
          (daily_bias_raw == "bullish" and side_lower == "buy") or
          (daily_bias_raw == "bearish" and side_lower == "sell")
        )
        if is_continuation:
          if allow_trend_aligned_short(
            daily_exhausted=daily_exhausted, daily_bias_raw=daily_bias_raw, side=side_lower,
            bias_1h=gate.get("intraday_bias_1h", "neutral"), bias_15m=gate.get("intraday_bias_15m", "neutral"),
            confidence=confidence, cfg=cfg.regime,
          ):
            logger.info("TREND-SHORT ALLOWED: futures limit %s %s — exhausted-bearish daily but 1h/15m confirm downtrend resumption (conf=%.2f)", side_lower, spot_symbol, confidence or 0.0)
          else:
            logger.warning("ANTI-FOMO BLOCK: futures limit %s %s rejected — daily %s exhausted", side_lower, spot_symbol, daily_bias_raw)
            return {"rejected": True, "reason": f"Daily exhaustion: {daily_bias_raw} trend overextended — no continuation entry", "hint": "Daily RSI is at an extreme. Wait for the pullback or trade counter-trend."}
      if daily_bias != "neutral" and not daily_exhausted:
        opposing = (daily_bias == "bearish" and side_lower == "buy") or (daily_bias == "bullish" and side_lower == "sell")
        if opposing:
          if allow_reversal_long(
            daily_bias=daily_bias, side=side_lower,
            bias_1h=gate.get("intraday_bias_1h", "neutral"), bias_15m=gate.get("intraday_bias_15m", "neutral"),
            confidence=confidence, cfg=cfg.regime,
          ):
            logger.info("REVERSAL LONG ALLOWED: futures limit %s %s — bearish daily but 1h/15m confirm a turn (conf=%.2f)", side_lower, spot_symbol, confidence or 0.0)
          else:
            return {"rejected": True, "reason": f"Daily gate: 1D trend is {daily_bias} — {side_lower} entry blocked", "hint": "Trade with the daily trend, take a confirmed reversal (1h+15m turned, high confidence), or switch symbol."}
      intraday_bias_1h_fl = gate.get("intraday_bias_1h", "neutral")
      intraday_1h_opposes_fl = (
        (intraday_bias_1h_fl == "bearish" and side_lower == "buy") or
        (intraday_bias_1h_fl == "bullish" and side_lower == "sell")
      )
      if intraday_1h_opposes_fl:
        if resolve_gate_deadlock(
          daily_bias=daily_bias, daily_exhausted=daily_exhausted, side=side_lower,
          bias_1h=intraday_bias_1h_fl, bias_15m=gate.get("intraday_bias_15m", "neutral"),
          confidence=confidence, cfg=cfg.regime,
        ):
          logger.info("DEADLOCK BREAK: futures limit %s %s — daily-aligned entry allowed past stalling 1h counter-bounce (daily=%s, 1h=%s, conf=%.2f)", side_lower, spot_symbol, daily_bias, intraday_bias_1h_fl, confidence or 0.0)
        else:
          logger.warning("1H ALIGN BLOCK: futures limit %s %s rejected — 1h bias %s opposes %s", side_lower, spot_symbol, intraday_bias_1h_fl, side_lower)
          return {"rejected": True, "reason": f"1h trend is {intraday_bias_1h_fl} — {side_lower} entry blocked", "hint": "1h timeframe opposes this direction. The daily trend is in correction. Wait for 1h alignment."}
      tf_conflict_fl = gate.get("timeframe_conflict", False)
      intraday_bias_15m_fl = gate.get("intraday_bias_15m", "neutral")
      if tf_conflict_fl and intraday_bias_15m_fl != "neutral":
        intraday_opposes_fl = (
          (intraday_bias_15m_fl == "bearish" and side_lower == "buy") or
          (intraday_bias_15m_fl == "bullish" and side_lower == "sell")
        )
        if intraday_opposes_fl:
          logger.warning("TF CONFLICT BLOCK: futures limit %s %s rejected — daily/intraday split, 15m %s opposes %s", side_lower, spot_symbol, intraday_bias_15m_fl, side_lower)
          return {"rejected": True, "reason": f"Timeframe conflict: 15m {intraday_bias_15m_fl} opposes proposed {side_lower}", "hint": "Wait for 15m to align with the higher-TF bias, or pick a different symbol."}

    # Correlation gate: block alt LONGs while BTC's daily regime is bearish (the RE-USDT failure mode).
    if block_alt_long_in_btc_downtrend(symbol=spot_symbol, side=side_lower, btc_daily_bias=_btc_daily_bias(), cfg=cfg.regime):
      logger.warning("CORRELATION GATE BLOCK: futures limit long %s rejected — BTC daily regime is bearish", spot_symbol)
      return {"rejected": True, "reason": f"Correlation gate: BTC daily regime is bearish — long on alt {spot_symbol} blocked", "hint": "Alts are high-beta to BTC; do NOT long an altcoin while BTC's daily trend is down. Trade a trend-aligned short, wait for BTC's daily to turn, or trade a major instead."}

    # Symbol bench: quarantine a repeatedly-losing symbol until its cooldown lifts.
    _bench_until_fl = _edge_state()["bench"].get(spot_symbol, 0)
    if _bench_until_fl > int(time.time()):
      _bench_hours_fl = (_bench_until_fl - int(time.time())) / 3600.0
      logger.warning("SYMBOL BENCH BLOCK: futures limit %s %s rejected — repeated recent losses, benched %.1fh more", side_lower, spot_symbol, _bench_hours_fl)
      return {"rejected": True, "reason": f"Symbol benched: {spot_symbol} lost repeatedly in its recent closes — no entries for {_bench_hours_fl:.1f}h more", "symbol": spot_symbol, "hint": "This symbol keeps stopping out; the setup that looks compliant is not working in the current tape. Trade a different symbol or stand aside — the bench lifts automatically."}

    _atr_scale_fl = 1.0
    if spot_symbol in _daily_gate_state:
      gate = _daily_gate_state[spot_symbol]
      daily_atr_pct = gate.get("daily_atr_pct")
      if daily_atr_pct is not None and daily_atr_pct > cfg.trading.max_atr_pct_for_entry:
        hard_limit = cfg.trading.max_atr_pct_for_entry * 1.5
        if daily_atr_pct > hard_limit:
          return {"rejected": True, "reason": f"Extreme volatility: ATR={daily_atr_pct:.2f}% > {hard_limit:.1f}% hard limit"}
        _atr_scale_fl = max(0.30, (cfg.trading.max_atr_pct_for_entry / daily_atr_pct) ** 2)
        logger.info("VOLATILITY SOFT GATE: futures limit %s ATR=%.2f%% — scaling position to %.0f%%", spot_symbol, daily_atr_pct, _atr_scale_fl * 100)

    # Regime throttle (B): shrink size + raise the confidence bar in a hostile (bearish/exhausted) daily.
    _g_fl = _daily_gate_state.get(spot_symbol, {})
    _atr_scale_fl = max(0.30, _atr_scale_fl * regime_size_factor(_g_fl.get("daily_bias", "neutral"), _g_fl.get("daily_exhausted", False), cfg.regime))

    if confidence is not None:
      _eff_min_fl = effective_min_confidence(cfg.trading.min_confidence, _g_fl.get("daily_bias", "neutral"), _g_fl.get("daily_exhausted", False), cfg.regime)
      if confidence < _eff_min_fl:
        return {"rejected": True, "reason": f"Confidence {confidence:.2f} below regime-adjusted minimum {_eff_min_fl:.2f}"}
      # Conviction sizing: shrink low-conviction entries, ramping to full size as conviction rises.
      _conv_scale_fl = conviction_size_factor(confidence, _eff_min_fl, cfg.regime)
      if _conv_scale_fl < 1.0:
        logger.info("CONVICTION SIZING: futures limit %s conf=%.2f (floor %.2f) — scaling position to %.0f%%", spot_symbol, confidence, _eff_min_fl, _conv_scale_fl * 100)
        _atr_scale_fl = max(0.30, _atr_scale_fl * _conv_scale_fl)

    # Loss-streak throttle: consecutive realized losses shrink the next entry (anti-martingale).
    _streak_fl = _edge_state()["size_factor"]
    if _streak_fl < 1.0:
      logger.info("LOSS-STREAK THROTTLE: futures limit %s — sizing to %.0f%% after %d consecutive losses", spot_symbol, _streak_fl * 100, _edge_state()["stats"].get("loss_streak", 0))
      _atr_scale_fl = max(0.30, _atr_scale_fl * _streak_fl)

    trades_today = memory.trades_today(spot_symbol)
    if trades_today >= cfg.trading.max_trades_per_symbol_per_day:
      return {"rejected": True, "reason": "Daily trade cap reached"}

    if cfg.trading.sentiment_filter_enabled:
      latest_sent = memory.latest_sentiment(symbol)
      day_key = int(time.time() // 86400)
      if not latest_sent or latest_sent.get("day") != day_key:
        return {"rejected": True, "reason": "Sentiment missing for today"}
      if latest_sent.get("score", 0) < cfg.trading.sentiment_min_score:
        return {"rejected": True, "reason": "Sentiment below threshold", "score": latest_sent.get("score")}

    futures_symbol = _to_futures_symbol(spot_symbol)
    if not futures_symbol:
      return {"error": "Invalid symbol for futures", "symbol": spot_symbol}
    contract = _get_contract_spec(futures_symbol)
    if not contract:
      return {"error": "Futures contract spec unavailable", "futuresSymbol": futures_symbol}

    # New-listing guard: skip freshly-listed, thin, ultra-volatile contracts (the RE-USDT trap).
    if cfg.trading.min_futures_listing_age_days > 0:
      _first_open = _to_float(contract.get("firstOpenDate"))
      _first_open_s = _first_open / 1000.0 if _first_open > 1e12 else _first_open
      if _first_open_s > 0:
        _age_days = (time.time() - _first_open_s) / 86400.0
        if _age_days < cfg.trading.min_futures_listing_age_days:
          logger.warning("NEW-LISTING BLOCK: %s rejected — contract age %.1fd < %.1fd minimum", futures_symbol, _age_days, cfg.trading.min_futures_listing_age_days)
          return {"rejected": True, "reason": f"New-listing guard: {futures_symbol} is {_age_days:.1f}d old (< {cfg.trading.min_futures_listing_age_days:.0f}d minimum)", "hint": "Freshly-listed perps are thin and ultra-volatile (this is how RE-USDT blew up). Wait for a real price history, or trade an established contract."}

    try:
      current_price = float(snapshot.tickers[spot_symbol].price)
    except Exception:
      return {"error": "Unable to fetch current price for deviation check"}
    if current_price <= 0:
      return {"error": "Invalid current price"}

    deviation = abs(entry_price_val - current_price) / current_price
    if deviation < cfg.trading.min_entry_deviation_pct:
      return {
        "rejected": True,
        "reason": f"entry_price {entry_price_val} is within {deviation*100:.3f}% of current price {current_price:.4f} (minimum deviation: {cfg.trading.min_entry_deviation_pct*100:.2f}%)",
        "hint": "Price is already at your target — use place_futures_market_order to enter immediately, or pick a further entry_price to genuinely wait for a better level.",
        "currentPrice": current_price,
        "entryPrice": entry_price_val,
      }

    multiplier = _to_float(contract.get("multiplier"))
    lot_size = int(contract.get("lotSize") or 1)
    max_order_qty = contract.get("maxOrderQty")
    contract_max_leverage = _to_float(contract.get("maxLeverage")) or None
    fee_rate = _to_float(contract.get("takerFeeRate")) or fees.get("futures_taker", 0.0006)
    slippage_rate = cfg.trading.estimated_slippage_pct
    if multiplier <= 0:
      return {"error": "Invalid contract multiplier", "contract": contract}

    leverage_caps = [c for c in (snapshot.max_leverage, cfg.trading.max_leverage, contract_max_leverage) if c and c > 0]
    if cfg.trading.max_entry_leverage > 0:
      leverage_caps.append(cfg.trading.max_entry_leverage)
    max_leverage_cap = min(leverage_caps) if leverage_caps else 3.0
    lev = min(max(1.0, lev_requested), max_leverage_cap)
    leverage_clamped = lev < lev_requested - 1e-9

    notional_input = float(notional_usd or 0) * _atr_scale_fl
    # Concentration cap: shrink notional to <= max_position_equity_pct of total equity (blast-radius guard).
    _conc_scale = concentration_scale(notional_input, snapshot.total_usdt, cfg.trading.max_position_equity_pct)
    if _conc_scale < 1.0:
      logger.info("CONCENTRATION CAP: futures limit %s notional %.2f → %.2f (<= %.0f%% of equity %.2f)",
                  spot_symbol, notional_input, notional_input * _conc_scale, cfg.trading.max_position_equity_pct * 100, snapshot.total_usdt)
      notional_input *= _conc_scale
    try:
      base_size = float(size_override) if size_override is not None else notional_input / entry_price_val
    except (TypeError, ValueError):
      return {"error": "Invalid size_override"}
    if base_size <= 0:
      return {"error": "Computed size is zero"}

    contracts_raw = base_size / multiplier
    lot = max(1, lot_size)
    contracts = int(math.ceil(contracts_raw / lot) * lot)
    actual_notional = contracts * multiplier * entry_price_val
    min_notional = lot * multiplier * entry_price_val
    if actual_notional < min_notional:
      return {"rejected": True, "reason": "Below contract minimum notional", "minNotionalUsd": min_notional}
    if max_order_qty and isinstance(max_order_qty, (int, float)) and contracts > max_order_qty:
      return {"rejected": True, "reason": "Exceeds max order size", "maxContracts": max_order_qty}
    if actual_notional > snapshot.max_position_usd * max_leverage_cap:
      return {"rejected": True, "reason": "Exceeds max notional cap", "cap": snapshot.max_position_usd * max_leverage_cap}

    # Reward:risk floor — when a bracket is supplied inline, reject entries that risk more than they
    # aim to make (the futures asymmetry fix; the deferred-bracket path is steered by the prompt).
    if cfg.trading.min_futures_rr > 0 and take_profit_price is not None and stop_loss_price is not None:
      _req_rr_fl = _edge_state()["required_rr"]
      _rr_fl = reward_risk_ratio(side_lower, entry_price_val, take_profit_price, stop_loss_price)
      if _rr_fl is None:
        logger.warning("RR BLOCK: futures limit %s %s rejected — TP/SL on the wrong side of entry", side_lower, spot_symbol)
        return {"rejected": True, "reason": "TP/SL on the wrong side of entry — cannot form a valid bracket", "entryPrice": entry_price_val, "takeProfit": take_profit_price, "stopLoss": stop_loss_price, "hint": "For a long, TP must be above and SL below entry; reverse for a short."}
      if _rr_fl < _req_rr_fl:
        logger.warning("RR BLOCK: futures limit %s %s rejected — reward:risk %.2f < %.2f", side_lower, spot_symbol, _rr_fl, _req_rr_fl)
        return {"rejected": True, "reason": f"Reward:risk {_rr_fl:.2f} below minimum {_req_rr_fl:.2f}", "rr": _rr_fl, "minRr": _req_rr_fl, "entryPrice": entry_price_val, "takeProfit": take_profit_price, "stopLoss": stop_loss_price, "hint": f"Widen the take-profit or tighten the stop so the TP distance is at least {_req_rr_fl:.1f}x the stop distance (floor rises while recent trades are net-losing). Skip setups that can't clear it."}

    expiry_ts = int(time.time()) + int(cfg.trading.entry_limit_expiry_minutes * 60)
    pending_protection = {"takeProfitPrice": take_profit_price, "stopLossPrice": stop_loss_price}
    note = (
      f"Futures limit {side_lower} placed at {entry_price_val} (current {current_price:.4f}, {deviation*100:.2f}% away). "
      f"Order expires in {cfg.trading.entry_limit_expiry_minutes:.0f}min if unfilled. "
      "On next run: check open positions — if filled, place bracket TP/SL with place_futures_stop_order."
    )

    def _build_limit_order(mode: str | None) -> KucoinFuturesOrderRequest:
      mode_norm = None
      auto_deposit = None
      if mode:
        mode_norm = mode.upper()
        if mode_norm == "ISOLATED":
          auto_deposit = False
      return KucoinFuturesOrderRequest(
        symbol=futures_symbol,
        side=side_lower,
        type="limit",
        leverage=str(int(lev)),
        size=str(contracts),
        price=str(entry_price_val),
        clientOid=str(uuid.uuid4()),
        marginMode=mode_norm,
        autoDeposit=auto_deposit,
        reduceOnly=False,
      )

    if snapshot.paper_trading:
      order_req = _build_limit_order(None)
      record = memory.record_trade(spot_symbol, side, actual_notional, paper=True, price=entry_price_val, size=contracts * multiplier, venue="futures")
      decision = None
      if confidence is not None:
        decision = memory.log_decision(spot_symbol, f"futures_{side_lower}_limit", float(confidence), rationale or "paper futures limit entry", paper=True)
      return {
        "paper": True,
        "pendingLimitEntry": True,
        "orderRequest": order_req.__dict__,
        "entryPrice": entry_price_val,
        "currentPrice": current_price,
        "deviationPct": round(deviation * 100, 3),
        "contracts": contracts,
        "futuresSymbol": futures_symbol,
        "appliedLeverage": lev,
        "leverageClampedFrom": lev_requested if leverage_clamped else None,
        "expiresAt": expiry_ts,
        "pendingProtection": pending_protection,
        "tradeRecord": record,
        "decisionLog": decision,
        "note": note,
      }

    # Detect margin mode from existing position if available
    margin_mode: str | None = None
    try:
      position_info = kucoin_futures.get_position(futures_symbol)
      if isinstance(position_info, dict) and "crossMode" in position_info:
        margin_mode = "cross" if position_info.get("crossMode") else "isolated"
    except Exception:
      pass
    # Honor the configured margin mode (default cross); "auto" preserves the existing position's mode.
    if _futures_margin_mode in ("cross", "isolated"):
      margin_mode = _futures_margin_mode
    elif margin_mode is None:
      margin_mode = "cross"

    order_req = _build_limit_order(margin_mode)
    try:
      kucoin_futures.set_margin_mode(futures_symbol, margin_mode)
    except Exception as exc:
      logger.warning("set_margin_mode failed (continuing): %s", exc)
    _apply_cross_leverage(futures_symbol, lev, margin_mode)
    try:
      res = kucoin_futures.place_order(order_req).__dict__
      if confidence is not None:
        res["decisionLog"] = memory.log_decision(spot_symbol, f"futures_{side_lower}_limit", float(confidence), rationale or "live futures limit entry", paper=False)
      res["pendingLimitEntry"] = True
      res["entryPrice"] = entry_price_val
      res["currentPrice"] = current_price
      res["deviationPct"] = round(deviation * 100, 3)
      res["contracts"] = contracts
      res["futuresSymbol"] = futures_symbol
      res["appliedLeverage"] = lev
      res["leverageClampedFrom"] = lev_requested if leverage_clamped else None
      res["expiresAt"] = expiry_ts
      res["pendingProtection"] = pending_protection
      res["note"] = note
      return res
    except Exception as exc:
      # Fallback: try opposite margin mode
      opposite_mode = "isolated" if margin_mode == "cross" else "cross"
      order_req = _build_limit_order(opposite_mode)
      try:
        kucoin_futures.set_margin_mode(futures_symbol, opposite_mode)
        _apply_cross_leverage(futures_symbol, lev, opposite_mode, context="fallback")
        res = kucoin_futures.place_order(order_req).__dict__
        if confidence is not None:
          res["decisionLog"] = memory.log_decision(spot_symbol, f"futures_{side_lower}_limit", float(confidence), rationale or "live futures limit entry", paper=False)
        res["pendingLimitEntry"] = True
        res["entryPrice"] = entry_price_val
        res["currentPrice"] = current_price
        res["deviationPct"] = round(deviation * 100, 3)
        res["contracts"] = contracts
        res["futuresSymbol"] = futures_symbol
        res["appliedLeverage"] = lev
        res["expiresAt"] = expiry_ts
        res["pendingProtection"] = pending_protection
        res["note"] = note
        return res
      except Exception as exc2:
        return {"error": f"Futures limit order failed: {exc2}", "firstAttemptError": str(exc), "futuresSymbol": futures_symbol}

  async def _place_futures_stop_order_impl(
    symbol: str,
    side: str,
    leverage: float,
    size: float | None = None,
    stop_price: float | None = None,
    stop: str | None = None,
    stop_price_type: str | None = None,
    take_profit_price: float | None = None,
    stop_loss_price: float | None = None,
    reduce_only: bool | None = None,
    close_order: bool | None = None,
    order_type: str = "limit",
    limit_price: float | None = None,
    client_oid: str | None = None,
  ) -> Dict[str, Any]:
    """Place a futures stop/TP/SL order (works for reduce-only hedges)."""
    requested_symbol = symbol
    symbol = _resolve_allowed_spot_symbol(symbol, allowed_symbols)
    if not symbol:
      symbol = _repair_allowed_symbol(requested_symbol)
    if not symbol:
      return {"error": "Unsupported symbol", "allowed": sorted(allowed_symbols), "requested": requested_symbol}
    if not cfg.kucoin_futures.enabled or not kucoin_futures:
      return {"error": "Futures disabled in config"}
    futures_symbol = _to_futures_symbol(symbol)
    if not futures_symbol:
      return {"error": "Invalid symbol for futures", "symbol": symbol}
    try:
      lev = float(leverage or 0)
    except Exception:
      return {"error": "Invalid leverage"}
    if lev <= 0 or lev > snapshot.max_leverage:
      lev = min(max(1.0, lev), snapshot.max_leverage)
    if size is None or size <= 0:
      return {"error": "size required and >0"}
    contract = _get_contract_spec(futures_symbol)
    contract_max_leverage = _to_float(contract.get("maxLeverage")) if contract else None
    if contract_max_leverage:
      lev = min(lev, contract_max_leverage)
    multiplier = _to_float(contract.get("multiplier")) if contract else None
    lot_size = int(contract.get("lotSize") or 1) if contract else 1
    if multiplier is None or multiplier <= 0:
      return {"error": "Invalid contract multiplier for stops", "contract": contract}
    contracts_raw = float(size) / multiplier
    lot = max(1, lot_size)
    contracts = int(math.ceil(contracts_raw / lot) * lot)
    if contracts <= 0:
      return {"error": "Computed contract size <=0", "contracts": contracts, "baseSize": size, "multiplier": multiplier}
    stop_price_str = f"{stop_price}" if stop_price is not None else None
    tp_str = f"{take_profit_price}" if take_profit_price is not None else None
    sl_str = f"{stop_loss_price}" if stop_loss_price is not None else None

    margin_mode: str | None = None
    try:
      position_info = kucoin_futures.get_position(futures_symbol)
      if isinstance(position_info, dict) and "crossMode" in position_info:
        margin_mode = "cross" if position_info.get("crossMode") else "isolated"
    except Exception as exc:
      logger.warning("Futures position lookup failed for stop order; margin mode unknown: %s", exc)
    if _futures_margin_mode in ("cross", "isolated"):
      margin_mode = _futures_margin_mode

    margin_mode_norm = margin_mode.upper() if margin_mode else None
    try:
      kucoin_futures.set_margin_mode(futures_symbol, margin_mode_norm or "CROSS")
    except Exception as exc:
      logger.warning("set_margin_mode failed for stop order (continuing): %s", exc)
    _apply_cross_leverage(futures_symbol, lev, margin_mode, context="stop order")
    order_req = KucoinFuturesOrderRequest(
      symbol=futures_symbol,
      side="buy" if side == "buy" else "sell",
      type="limit" if order_type == "limit" else "market",
      leverage=str(int(lev)),
      size=f"{contracts}",
      price=f"{limit_price}" if limit_price is not None else None,
      clientOid=client_oid or str(uuid.uuid4()),
      stop=stop,
      stopPriceType=stop_price_type,
      stopPrice=stop_price_str,
      reduceOnly=reduce_only,
      closeOrder=close_order,
      takeProfitPrice=tp_str,
      stopLossPrice=sl_str,
      marginMode=margin_mode_norm,
      autoDeposit=False if margin_mode_norm == "ISOLATED" else None,
    )
    if snapshot.paper_trading:
      return {"paper": True, "orderRequest": order_req.__dict__}
    try:
      res = kucoin_futures.place_order(order_req).__dict__
      res["orderRequest"] = order_req.__dict__
      return res
    except Exception as exc:
      return {"error": str(exc), "orderRequest": order_req.__dict__}

  @function_tool
  async def place_futures_stop_order(
    symbol: str,
    side: str,
    leverage: float,
    size: float | None = None,
    stop_price: float | None = None,
    stop: str | None = None,
    stop_price_type: str | None = None,
    take_profit_price: float | None = None,
    stop_loss_price: float | None = None,
    reduce_only: bool | None = None,
    close_order: bool | None = None,
    order_type: str = "limit",
    limit_price: float | None = None,
    client_oid: str | None = None,
  ) -> Dict[str, Any]:
    """Place a futures stop/TP/SL order (works for reduce-only hedges)."""
    return await _place_futures_stop_order_impl(
      symbol=symbol,
      side=side,
      leverage=leverage,
      size=size,
      stop_price=stop_price,
      stop=stop,
      stop_price_type=stop_price_type,
      take_profit_price=take_profit_price,
      stop_loss_price=stop_loss_price,
      reduce_only=reduce_only,
      close_order=close_order,
      order_type=order_type,
      limit_price=limit_price,
      client_oid=client_oid,
    )

  @function_tool
  async def cancel_futures_order(order_id: str, symbol: str | None = None) -> Dict[str, Any]:
    """Cancel a futures order (stop or regular) by orderId."""
    if snapshot.paper_trading:
      return {"paper": True, "cancelled": {"orderId": order_id, "symbol": symbol}}
    if not cfg.kucoin_futures.enabled or not kucoin_futures:
      return {"error": "Futures disabled in config"}
    try:
      res = kucoin_futures.cancel_order(order_id, symbol=symbol)
      return {"cancelled": res, "orderId": order_id, "symbol": symbol}
    except Exception as exc:
      return {"error": str(exc), "orderId": order_id, "symbol": symbol}

  @function_tool
  async def list_futures_stop_orders(status: str = "active", symbol: str | None = None) -> Dict[str, Any]:
    """List futures stop orders; status: active/done."""
    if not cfg.kucoin_futures.enabled or not kucoin_futures:
      return {"error": "Futures disabled in config"}
    try:
      orders = kucoin_futures.list_stop_orders(status=status, symbol=symbol)
      return {"orders": orders, "status": status, "symbol": symbol}
    except Exception as exc:
      return {"error": str(exc), "status": status, "symbol": symbol}

  @function_tool
  async def list_futures_positions(status: str | None = None) -> Dict[str, Any]:
    """List current futures positions (live fetch; falls back to snapshot)."""
    if not cfg.kucoin_futures.enabled or not kucoin_futures:
      return {"error": "Futures disabled in config"}
    try:
      positions = kucoin_futures.list_positions(status=status)
      return {"positions": positions, "status": status}
    except Exception as exc:
      return {"error": str(exc), "status": status, "snapshotPositions": snapshot.futures_positions}


  # ── Fills & closed positions ────────────────────────────────────────────────────────────────────
  @function_tool
  async def get_recent_fills(venue: str = "all", symbol: str | None = None) -> Dict[str, Any]:
    """Get recent trade fills (executions) including those from triggered stop orders. venue: 'spot', 'futures', or 'all'."""
    result: Dict[str, Any] = {}
    resolved = _resolve_allowed_spot_symbol(symbol, allowed_symbols) if symbol else None
    if venue in ("all", "spot"):
      try:
        spot_fills = kucoin.get_fills(symbol=resolved, page_size=30)
        result["spotFills"] = spot_fills
      except Exception as exc:
        result["spotFillsError"] = str(exc)
    if venue in ("all", "futures"):
      if not cfg.kucoin_futures.enabled or not kucoin_futures:
        result["futuresFillsError"] = "Futures disabled"
      else:
        futures_sym = _to_futures_symbol(resolved) if resolved else None
        try:
          futures_fills = kucoin_futures.get_fills(symbol=futures_sym, page_size=30)
          result["futuresFills"] = futures_fills
        except Exception as exc:
          result["futuresFillsError"] = str(exc)
    return result

  @function_tool
  async def get_closed_positions(symbol: str | None = None) -> Dict[str, Any]:
    """Get closed futures positions with realized PnL. Shows positions that were closed (by TP/SL triggers, manual close, or liquidation)."""
    if not cfg.kucoin_futures.enabled or not kucoin_futures:
      return {"error": "Futures disabled in config"}
    futures_sym = None
    if symbol:
      resolved = _resolve_allowed_spot_symbol(symbol, allowed_symbols)
      futures_sym = _to_futures_symbol(resolved) if resolved else None
    try:
      closed = kucoin_futures.get_position_history(symbol=futures_sym, page_size=20)
      return {"closedPositions": closed}
    except Exception as exc:
      return {"error": str(exc)}

  _active_contracts_cache: Dict[str, Any] = {"ts": 0.0, "symbols": None}

  def _active_contract_symbols() -> set[str] | None:
    """Set of live KuCoin perpetual symbols, cached 1h. None = lookup unavailable (fail open)."""
    now = time.time()
    if _active_contracts_cache["symbols"] is not None and now - _active_contracts_cache["ts"] < 3600:
      return _active_contracts_cache["symbols"]
    if not kucoin_futures:
      return None
    try:
      contracts = kucoin_futures.list_active_contracts()
      symbols = {str(c.get("symbol") or "").strip().upper() for c in contracts if isinstance(c, dict)}
      symbols.discard("")
      if symbols:
        _active_contracts_cache["symbols"] = symbols
        _active_contracts_cache["ts"] = now
        return symbols
    except Exception as exc:
      logger.debug("Active-contract list unavailable (symbol validation fails open): %s", exc)
    return _active_contracts_cache["symbols"]

  def _resolve_futures_symbol(symbol: str) -> tuple[str | None, str | None]:
    """Resolve flexible symbol input to a futures contract symbol. Returns (futures_sym, error_msg).

    Raw '*USDTM' input used to be passed to the API verbatim, so malformed variants
    ('SOL-USDTM', 'BTCUSDTM' — KuCoin's BTC perp is XBTUSDTM) burned doomed API calls that
    surfaced as opaque '200003 Invalid symbol' / '404000' decision errors. Normalize first,
    then validate against the live contract list (fail-open if the list is unavailable).
    """
    upper = (symbol or "").upper().strip().replace("-", "").replace("/", "").replace("_", "")
    if upper.endswith("USDTM"):
      if upper.startswith("BTC"):
        upper = "XBT" + upper[3:]
      active = _active_contract_symbols()
      if active is not None and upper not in active:
        return None, f"No active KuCoin perpetual named '{upper}'"
      return upper, None
    spot = _resolve_allowed_spot_symbol(symbol, allowed_symbols)
    if not spot:
      spot = _repair_allowed_symbol(symbol)
    if not spot:
      return None, f"Unknown symbol '{symbol}'"
    fsym = _to_futures_symbol(spot)
    return fsym, None if fsym else f"Cannot map '{symbol}' to futures"


  # ── Futures market data & whole-universe screening ──────────────────────────────────────────────
  @function_tool
  async def fetch_funding_rate(symbol: str, history_hours: int = 0) -> Dict[str, Any]:
    """Get current funding rate (and predicted next). Set history_hours > 0 to include recent funding rate history."""
    if not cfg.kucoin_futures.enabled or not kucoin_futures:
      return {"error": "Futures disabled in config"}
    fsym, err = _resolve_futures_symbol(symbol)
    if err:
      return {"error": err, "requested": symbol}
    result: Dict[str, Any] = {}
    try:
      result["current"] = kucoin_futures.get_funding_rate(fsym)
    except Exception as exc:
      result["currentError"] = str(exc)
    if history_hours > 0:
      history_hours = min(history_hours, 168)
      end_ms = kucoin._timestamp_ms()
      start_ms = end_ms - history_hours * 3600 * 1000
      try:
        result["history"] = kucoin_futures.get_funding_rate_history(fsym, start_at=start_ms, end_at=end_ms)
      except Exception as exc:
        result["historyError"] = str(exc)
    result["symbol"] = fsym
    return result

  @function_tool
  async def fetch_open_interest(symbol: str) -> Dict[str, Any]:
    """Get current open interest, mark price, index price, and key contract specs for a futures symbol."""
    if not cfg.kucoin_futures.enabled or not kucoin_futures:
      return {"error": "Futures disabled in config"}
    fsym, err = _resolve_futures_symbol(symbol)
    if err:
      return {"error": err, "requested": symbol}
    try:
      contract = kucoin_futures.get_contract_detail(fsym)
      return {
        "symbol": fsym,
        "openInterest": contract.get("openInterest"),
        "markPrice": contract.get("markPrice"),
        "indexPrice": contract.get("indexPrice"),
        "lastTradePrice": contract.get("lastTradePrice"),
        "fundingFeeRate": contract.get("fundingFeeRate"),
        "predictedFundingFeeRate": contract.get("predictedFundingFeeRate"),
        "turnoverOf24h": contract.get("turnoverOf24h"),
        "volumeOf24h": contract.get("volumeOf24h"),
      }
    except Exception as exc:
      return {"error": str(exc), "symbol": fsym}

  @function_tool
  async def scan_futures_market(top_n: int = 15, sort_by: str = "momentum", side: str = "both") -> Dict[str, Any]:
    """Screen the ENTIRE KuCoin USDT-perpetual universe for tradable opportunities RIGHT NOW.

    DISCOVERY tool — the only one that sees the whole market instead of one named symbol. Ranks all
    active perps so you can find the coin that is actually moving, beyond the current list. It pre-applies
    the same code-level quality bars the entry gates enforce (a 24h-turnover liquidity floor and the
    minimum listing age), so every result is liquid and mature enough to consider — no fresh micro-caps.
    Use it every research run: scan, then validate the best candidates with analyze_market_context /
    fetch_orderbook before add_coin.

    Args:
      top_n: how many to return (max 40).
      sort_by: 'momentum' (largest absolute 24h move, default), 'gainers', 'losers', or 'volume'.
      side: 'both' (default), 'long' (up movers only) or 'short' (down movers only). In a bearish BTC
            daily, alt LONGs are still blocked by the correlation gate — prefer 'short' or majors then.
    """
    if not cfg.kucoin_futures.enabled or not kucoin_futures:
      return {"error": "Futures disabled in config"}
    try:
      contracts = kucoin_futures.list_active_contracts()
    except Exception as exc:
      return {"error": f"Could not fetch active contracts: {exc}"}
    screened = _screen_contracts(
      contracts,
      min_turnover=cfg.trading.screener_min_turnover_usd_24h,
      min_age_days=cfg.trading.min_futures_listing_age_days,
      now=time.time(),
      sort_by=sort_by,
      side=side,
      top_n=top_n,
    )
    screened.update({
      "minTurnoverUsd": cfg.trading.screener_min_turnover_usd_24h,
      "minAgeDays": cfg.trading.min_futures_listing_age_days,
      "sortBy": sort_by,
      "side": side,
      "note": (
        "Results cleared the liquidity + listing-age bars. Validate a candidate with "
        "analyze_market_context/fetch_orderbook before add_coin. Entry gates (daily/1h/RR/correlation) "
        "still apply at trade time — in a bearish BTC daily, alt LONGs stay blocked, so prefer shorts or majors."
      ),
    })
    return screened

  @function_tool
  async def fetch_futures_orderbook(symbol: str, depth: int = 20) -> Dict[str, Any]:
    """Fetch futures level2 orderbook snapshot (depth 20 or 100). Use to assess futures-specific liquidity and support/resistance."""
    if not cfg.kucoin_futures.enabled or not kucoin_futures:
      return {"error": "Futures disabled in config"}
    fsym, err = _resolve_futures_symbol(symbol)
    if err:
      return {"error": err, "requested": symbol}
    depth_safe = 20 if depth <= 20 else 100
    try:
      ob = kucoin_futures.get_orderbook(fsym, depth=depth_safe)
      ob["bids"] = (ob.get("bids") or [])[:depth_safe]
      ob["asks"] = (ob.get("asks") or [])[:depth_safe]
      return {"symbol": fsym, "depth": depth_safe, "orderbook": ob}
    except Exception as exc:
      return {"error": str(exc), "symbol": fsym}

  @function_tool
  async def fetch_futures_mark_price(symbol: str) -> Dict[str, Any]:
    """Get current mark price and index price for a futures contract. The basis (mark - index) reveals futures premium/discount and market sentiment."""
    if not cfg.kucoin_futures.enabled or not kucoin_futures:
      return {"error": "Futures disabled in config"}
    fsym, err = _resolve_futures_symbol(symbol)
    if err:
      return {"error": err, "requested": symbol}
    try:
      data = kucoin_futures.get_mark_price(fsym)
      mark = float(data.get("value") or 0)
      index = float(data.get("indexPrice") or 0)
      basis = mark - index if mark and index else None
      basis_pct = (basis / index * 100) if basis is not None and index else None
      return {
        "symbol": fsym,
        "markPrice": mark,
        "indexPrice": index,
        "basis": round(basis, 6) if basis is not None else None,
        "basisPct": round(basis_pct, 4) if basis_pct is not None else None,
        "timePoint": data.get("timePoint"),
      }
    except Exception as exc:
      return {"error": str(exc), "symbol": fsym}

  @function_tool
  async def fetch_futures_candles(
    symbol: str,
    interval: str = "15min",
    lookback_minutes: int = 360,
  ) -> Dict[str, Any]:
    """Fetch futures OHLCV candles. Use to compare futures price action vs spot. Intervals: 1min, 5min, 15min, 30min, 1hour, 2hour, 4hour, 8hour, 12hour, 1day, 1week."""
    if not cfg.kucoin_futures.enabled or not kucoin_futures:
      return {"error": "Futures disabled in config"}
    fsym, err = _resolve_futures_symbol(symbol)
    if err:
      return {"error": err, "requested": symbol}
    granularity_map = {
      "1min": 1, "5min": 5, "15min": 15, "30min": 30,
      "1hour": 60, "2hour": 120, "4hour": 240, "8hour": 480,
      "12hour": 720, "1day": 1440, "1week": 10080,
    }
    gran = granularity_map.get(interval)
    if gran is None:
      return {"error": "Invalid interval", "allowed": list(granularity_map.keys())}
    lookback_min = max(1, min(int(lookback_minutes or 0), 10080))
    end_ms = kucoin._timestamp_ms()
    points = min(500, max(1, lookback_min // gran))
    start_ms = end_ms - points * gran * 60 * 1000
    try:
      candles = kucoin_futures.get_candles(fsym, granularity=gran, start_at=start_ms, end_at=end_ms)
      return {
        "symbol": fsym,
        "interval": interval,
        "points": candles[:500],
        "rows": len(candles),
        "note": "Each row: [time_ms, open, high, low, close, volume, turnover]",
      }
    except Exception as exc:
      return {"error": str(exc), "symbol": fsym}

  @function_tool
  async def fetch_contract_details(symbol: str) -> Dict[str, Any]:
    """Get full contract specification: multiplier, lotSize, tickSize, maxLeverage, maxOrderQty, takerFee, makerFee, funding interval, settlement currency, etc."""
    if not cfg.kucoin_futures.enabled or not kucoin_futures:
      return {"error": "Futures disabled in config"}
    fsym, err = _resolve_futures_symbol(symbol)
    if err:
      return {"error": err, "requested": symbol}
    try:
      return {"symbol": fsym, "contract": kucoin_futures.get_contract_detail(fsym)}
    except Exception as exc:
      return {"error": str(exc), "symbol": fsym}

  @function_tool
  async def set_futures_position_protection(
    symbol: str,
    take_profit_price: float | None = None,
    stop_loss_price: float | None = None,
    stop_price_type: str = "MP",
    cancel_existing: bool = True,
  ) -> Dict[str, Any]:
    """Add or replace TP/SL for an existing open futures position using reduce-only close orders."""
    requested_symbol = symbol
    symbol = _resolve_allowed_spot_symbol(symbol, allowed_symbols)
    if not symbol:
      symbol = _repair_allowed_symbol(requested_symbol)
    if not symbol:
      return {"error": "Unsupported symbol", "allowed": sorted(allowed_symbols), "requested": requested_symbol}
    if not cfg.kucoin_futures.enabled or not kucoin_futures:
      return {"error": "Futures disabled in config"}

    futures_symbol = _to_futures_symbol(symbol)
    if not futures_symbol:
      return {"error": "Invalid symbol for futures", "symbol": symbol}

    tp_val = _to_float(take_profit_price)
    sl_val = _to_float(stop_loss_price)
    if not tp_val and not sl_val:
      return {"error": "At least one of take_profit_price or stop_loss_price is required"}

    price_type = (stop_price_type or "MP").upper()
    if price_type not in {"TP", "MP", "IP"}:
      return {"error": "Invalid stop_price_type", "allowed": ["IP", "MP", "TP"]}

    try:
      position = kucoin_futures.get_position(futures_symbol)
    except Exception as exc:
      return {"error": f"Position lookup failed: {exc}", "symbol": symbol, "futuresSymbol": futures_symbol}

    if not isinstance(position, dict):
      return {"error": "Unexpected position payload", "position": position, "symbol": symbol, "futuresSymbol": futures_symbol}

    current_qty = _to_float(position.get("currentQty"))
    if abs(current_qty) <= 0:
      return {"error": "No open futures position", "symbol": symbol, "futuresSymbol": futures_symbol, "position": position}

    contract = _get_contract_spec(futures_symbol)
    multiplier = _to_float(contract.get("multiplier")) if contract else None
    if multiplier is None or multiplier <= 0:
      return {"error": "Invalid contract multiplier for protection", "contract": contract, "futuresSymbol": futures_symbol}

    base_size = abs(current_qty) * multiplier
    if base_size <= 0:
      return {"error": "Computed protection size <= 0", "currentQty": current_qty, "multiplier": multiplier}

    leverage_val = _to_float(position.get("realLeverage")) or _to_float(position.get("leverage")) or snapshot.max_leverage
    if leverage_val <= 0:
      leverage_val = snapshot.max_leverage or 1.0

    exit_side = "sell" if current_qty > 0 else "buy"
    tp_stop = "up" if current_qty > 0 else "down"
    sl_stop = "down" if current_qty > 0 else "up"

    cancelled: list[Dict[str, Any]] = []
    cancel_errors: list[Dict[str, Any]] = []
    if cancel_existing:
      try:
        active_orders = kucoin_futures.list_stop_orders(status="active", symbol=futures_symbol) or []
      except Exception as exc:
        active_orders = []
        cancel_errors.append({"stage": "list", "error": str(exc)})
      for order in active_orders:
        if not isinstance(order, dict):
          continue
        order_side = str(order.get("side") or "").lower()
        if order_side and order_side != exit_side:
          continue
        order_id = str(order.get("id") or order.get("orderId") or "").strip()
        if not order_id:
          continue
        if snapshot.paper_trading:
          cancelled.append({"paper": True, "orderId": order_id, "symbol": futures_symbol})
          continue
        try:
          res = kucoin_futures.cancel_order(order_id, symbol=futures_symbol)
          cancelled.append({"orderId": order_id, "response": res})
        except Exception as exc:
          cancel_errors.append({"orderId": order_id, "error": str(exc)})

    bracket: Dict[str, Any] = {}
    if sl_val:
      bracket["stopLoss"] = await _place_futures_stop_order_impl(
        symbol=symbol,
        side=exit_side,
        leverage=leverage_val,
        size=base_size,
        stop_price=sl_val,
        stop=sl_stop,
        stop_price_type=price_type,
        reduce_only=True,
        close_order=True,
        order_type="market",
        client_oid=f"{futures_symbol.lower()}-sl-{uuid.uuid4().hex[:12]}",
      )
    if tp_val:
      if cfg.trading.partial_tp_enabled and base_size > 0:
        contract_spec = _get_contract_spec(futures_symbol) or {}
        lot_size = int(contract_spec.get("lotSize") or 1)
        cur_price = float(snapshot.tickers.get(symbol, next(iter(snapshot.tickers.values()))).price) if snapshot.tickers else 0.0
        tp_distance = abs(tp_val - cur_price) if cur_price > 0 else 0
        tp1_price = cur_price + tp_distance * 0.6 * (1 if current_qty > 0 else -1) if tp_distance > 0 else tp_val
        tp2_price = tp_val
        raw_contracts = abs(current_qty)
        size_t1 = int(raw_contracts * 0.6 / lot_size) * lot_size
        size_t2 = int((raw_contracts - size_t1) / lot_size) * lot_size
        tranches_placed = []
        if size_t1 > 0 and ((current_qty > 0 and tp1_price > cur_price) or (current_qty < 0 and tp1_price < cur_price)):
          size_t1_base = size_t1 * multiplier
          res1 = await _place_futures_stop_order_impl(
            symbol=symbol, side=exit_side, leverage=leverage_val, size=size_t1_base,
            stop_price=tp1_price, stop=tp_stop, stop_price_type=price_type,
            reduce_only=True, close_order=False, order_type="market",
            client_oid=f"{futures_symbol.lower()}-tp1-{uuid.uuid4().hex[:12]}",
          )
          tranches_placed.append({"tranche": 1, "contracts": size_t1, "price": tp1_price, "result": res1})
        if size_t2 > 0 and ((current_qty > 0 and tp2_price > cur_price) or (current_qty < 0 and tp2_price < cur_price)):
          size_t2_base = size_t2 * multiplier
          res2 = await _place_futures_stop_order_impl(
            symbol=symbol, side=exit_side, leverage=leverage_val, size=size_t2_base,
            stop_price=tp2_price, stop=tp_stop, stop_price_type=price_type,
            reduce_only=True, close_order=True, order_type="market",
            client_oid=f"{futures_symbol.lower()}-tp2-{uuid.uuid4().hex[:12]}",
          )
          tranches_placed.append({"tranche": 2, "contracts": size_t2, "price": tp2_price, "result": res2})
        if tranches_placed:
          bracket["takeProfit"] = {"staged": True, "tranches": tranches_placed}
        else:
          bracket["takeProfit"] = await _place_futures_stop_order_impl(
            symbol=symbol, side=exit_side, leverage=leverage_val, size=base_size,
            stop_price=tp_val, stop=tp_stop, stop_price_type=price_type,
            reduce_only=True, close_order=True, order_type="market",
            client_oid=f"{futures_symbol.lower()}-tp-{uuid.uuid4().hex[:12]}",
          )
      else:
        bracket["takeProfit"] = await _place_futures_stop_order_impl(
          symbol=symbol,
          side=exit_side,
          leverage=leverage_val,
          size=base_size,
          stop_price=tp_val,
          stop=tp_stop,
          stop_price_type=price_type,
          reduce_only=True,
          close_order=True,
          order_type="market",
          client_oid=f"{futures_symbol.lower()}-tp-{uuid.uuid4().hex[:12]}",
        )

    return {
      "symbol": symbol,
      "futuresSymbol": futures_symbol,
      "positionSide": "long" if current_qty > 0 else "short",
      "positionQty": current_qty,
      "protectionSize": base_size,
      "exitSide": exit_side,
      "cancelledOrders": cancelled,
      "cancelErrors": cancel_errors,
      "bracket": bracket,
      "position": position,
    }


  # ── Transfers & account state ───────────────────────────────────────────────────────────────────
  @function_tool
  async def transfer_funds(
    direction: str,
    currency: str = "USDT",
    amount: float = 0.0,
  ) -> Dict[str, Any]:
    """Transfer funds between spot (trade), futures (contract), and financial (earn/pool)."""
    dir_norm = (direction or "").lower()
    allowed_dirs = {
      "spot_to_futures",
      "futures_to_spot",
      "financial_to_spot",
      "spot_to_financial",
      "financial_to_futures",
      "futures_to_financial",
    }
    if dir_norm not in allowed_dirs:
      return {"error": "Invalid direction", "allowed": sorted(allowed_dirs)}
    try:
      amt = float(amount or 0)
    except (TypeError, ValueError):
      return {"error": "Invalid amount"}
    if amt <= 0:
      return {"error": "Amount must be positive"}
    amt = math.floor(amt * 1e8) / 1e8

    # Pull fresh balances to avoid stale state.
    spot_accounts = kucoin.get_trade_accounts()
    spot_balance_map: Dict[str, float] = {}
    for bal in spot_accounts:
      spot_balance_map[bal.currency] = spot_balance_map.get(bal.currency, 0.0) + float(bal.available or 0)

    financial_balance_map: Dict[str, float] = {}
    try:
      financial_accounts = kucoin.get_financial_accounts()
      for bal in financial_accounts:
        financial_balance_map[bal.currency] = financial_balance_map.get(bal.currency, 0.0) + float(bal.available or 0)
    except Exception as exc:
      # Keep going; availability checks will fail gracefully if needed.
      financial_accounts = []
      financial_balance_map = {}

    futures_available = 0.0
    futures_overview: Dict[str, Any] | None = None
    if cfg.kucoin_futures.enabled and kucoin_futures:
      try:
        futures_overview = kucoin_futures.get_account_overview()
        futures_available = _to_float(futures_overview.get("availableBalance"))
      except Exception as exc:
        # Record but continue to avoid hard failure.
        futures_overview = {"error": str(exc)}

    route_map = {
      "spot_to_futures": ("trade", "contract"),
      "futures_to_spot": ("contract", "trade"),
      "financial_to_spot": ("financial", "trade"),
      "spot_to_financial": ("trade", "financial"),
      "financial_to_futures": ("financial", "contract"),
      "futures_to_financial": ("contract", "financial"),
    }
    from_acct, to_acct = route_map[dir_norm]
    cur = currency.upper()

    if from_acct == "trade":
      available = spot_balance_map.get(cur, 0.0)
      if amt > available:
        return {"rejected": True, "reason": "Insufficient spot balance", "available": available}
    elif from_acct == "financial":
      available = financial_balance_map.get(cur, 0.0)
      if amt > available:
        return {"rejected": True, "reason": "Insufficient financial balance", "available": available}
    elif from_acct == "contract":
      if not cfg.kucoin_futures.enabled or not kucoin_futures:
        return {"error": "Futures disabled or client unavailable"}
      available = futures_available
      if amt > available:
        return {
          "rejected": True,
          "reason": "Insufficient futures balance",
          "available": available,
          "futuresOverview": futures_overview,
        }

    if snapshot.paper_trading:
      return {
        "paper": True,
        "direction": dir_norm,
        "currency": currency.upper(),
        "amount": amt,
        "from": from_acct,
        "to": to_acct,
        "spotAvailable": spot_balance_map.get(cur, 0.0),
        "financialAvailable": financial_balance_map.get(cur, 0.0),
        "futuresAvailable": futures_available,
      }

    try:
      # Financial/Earn balances often require redemption before they are transferable.
      if dir_norm in {"financial_to_spot", "financial_to_futures"}:
        redeem_res = kucoin.redeem_earn_for_currency(cur, amt)
        if redeem_res.get("error"):
          return {
            "error": redeem_res.get("error"),
            "detail": redeem_res,
            "direction": dir_norm,
            "currency": currency.upper(),
            "amount": amt,
          }

        # After redeem, attempt to ensure funds are in trade (spot) before any futures transfer.
        main_accounts = kucoin.get_accounts("main")
        trade_accounts = kucoin.get_trade_accounts()
        main_available = sum(float(a.available or 0) for a in main_accounts if a.currency == cur)
        trade_available = sum(float(a.available or 0) for a in trade_accounts if a.currency == cur)

        if trade_available < amt and main_available >= amt:
          kucoin.transfer_funds(currency=cur, amount=amt, from_account="main", to_account="trade")
          trade_available = sum(float(a.available or 0) for a in kucoin.get_trade_accounts() if a.currency == cur)

        if dir_norm == "financial_to_spot":
          return {
            "transfer": {"redeem": redeem_res, "spotAvailableAfter": trade_available},
            "method": "redeem->spot",
            "direction": dir_norm,
            "currency": currency.upper(),
            "amount": amt,
            "spotAvailable": trade_available,
            "financialAvailable": financial_balance_map.get(cur, 0.0),
            "futuresAvailable": futures_available,
          }

        # financial_to_futures: move from trade to futures after redeem.
        step2 = kucoin.transfer_funds(currency=cur, amount=amt, from_account="trade", to_account="contract")
        return {
          "transfer": {"redeem": redeem_res, "trade_to_futures": step2},
          "method": "redeem->spot->futures",
          "direction": dir_norm,
          "currency": currency.upper(),
          "amount": amt,
          "spotAvailable": trade_available,
          "financialAvailable": financial_balance_map.get(cur, 0.0),
          "futuresAvailable": futures_available,
          "futuresOverview": futures_overview,
        }

      if dir_norm in {"spot_to_financial", "futures_to_financial"}:
        return {
          "error": "Transfers into Earn/Financial are not supported via API. Subscribe to Earn products directly.",
          "direction": dir_norm,
          "currency": currency.upper(),
          "amount": amt,
        }

      res = kucoin.transfer_funds(
        currency=currency.upper(),
        amount=amt,
        from_account=from_acct,  # type: ignore[arg-type]
        to_account=to_acct,  # type: ignore[arg-type]
      )
      return {
        "transfer": res,
        "method": "universal-transfer",
        "direction": dir_norm,
        "currency": currency.upper(),
        "amount": amt,
        "spotAvailable": spot_balance_map.get(cur, 0.0),
        "financialAvailable": financial_balance_map.get(cur, 0.0),
        "futuresAvailable": futures_available,
        "futuresOverview": futures_overview,
      }
    except Exception as exc:
      key_hint = ""
      try:
        key_hint = (kucoin.api_key or "")[-6:]
      except Exception:
        key_hint = ""
      return {
        "error": str(exc),
        "direction": dir_norm,
        "currency": currency.upper(),
        "amount": amt,
        "apiKeySuffix": key_hint or None,
        "baseUrl": getattr(kucoin, "base_url", None),
      }

  @function_tool
  async def fetch_account_state() -> Dict[str, Any]:
    """Get up-to-date balances for all spot accounts and futures (including non-trade funds)."""
    try:
      spot_accounts_fresh = kucoin.get_trade_accounts()
      main_accounts_fresh = kucoin.get_accounts("main")
      fin_accounts_fresh = kucoin.get_financial_accounts()
    except Exception as exc:
      return {"error": f"account_fetch_failed: {exc}"}

    futures_overview = None
    futures_error = None
    if cfg.kucoin_futures.enabled and kucoin_futures:
      try:
        futures_overview = kucoin_futures.get_account_overview()
      except Exception as exc:
        futures_error = f"futures_fetch_failed: {exc}"

    spot_totals = _aggregate_account_totals(spot_accounts_fresh)
    main_totals = _aggregate_account_totals(main_accounts_fresh)
    fin_totals = _aggregate_account_totals(fin_accounts_fresh)
    return _build_compact_account_state(
      spot_totals=spot_totals,
      funding_totals=main_totals,
      financial_totals=fin_totals,
      futures_overview=futures_overview,
      futures_enabled=bool(cfg.kucoin_futures.enabled),
      futures_error=futures_error,
    )

  @function_tool
  async def refresh_fee_rates() -> Dict[str, Any]:
    """Fetch latest fee rates (spot base fee). Futures fee not provided by API; keep previous/default."""
    try:
      base_fee = kucoin.get_base_fee()
      spot_taker = float(base_fee.get("takerFeeRate") or 0.001)
      spot_maker = float(base_fee.get("makerFeeRate") or 0.001)
    except Exception as exc:
      return {"error": f"base_fee_failed: {exc}"}
    entry = memory.save_fee_info(
      spot_taker=spot_taker,
      spot_maker=spot_maker,
      futures_taker=fees.get("futures_taker"),
      futures_maker=fees.get("futures_maker"),
    )
    return {"fee": entry}


  # ── Planning, memory & coin-list curation ───────────────────────────────────────────────────────
  @function_tool
  async def save_trade_plan(title: str, summary: str, actions: List[str]) -> Dict[str, Any]:
    """Persist a trading plan (title, summary, actions) for recall."""
    return memory.save_plan(title=title, summary=summary, actions=actions, author="Trading Agent")

  @function_tool
  async def latest_plan() -> Dict[str, Any]:
    """Get the latest stored trading plan."""
    return {"latest_plan": memory.latest_plan()}

  @function_tool
  async def latest_items(kind: str, limit: int = 5) -> Dict[str, Any]:
    """Fetch latest N memory entries (plans/research/sentiments/decisions/trades/triggers/coins/fees)."""
    if not kind:
      return {"error": "kind required"}
    return memory.latest_items(kind, limit)

  @function_tool
  async def clear_plans() -> Dict[str, Any]:
    """Clear all stored plans and triggers."""
    return memory.clear_plans()

  @function_tool
  async def set_auto_trigger(
    symbol: str,
    direction: str,
    rationale: str,
    target_price: float | None = None,
    stop_price: float | None = None,
  ) -> Dict[str, Any]:
    """Store an auto-buy/sell trigger idea (persists to disk for follow-up by future runs)."""
    symbol = _normalize_symbol(symbol)
    if symbol not in allowed_symbols:
      return {"error": "Unsupported symbol", "allowed": sorted(allowed_symbols)}
    return memory.save_trigger(
      symbol=symbol,
      direction=direction,
      rationale=rationale,
      target_price=target_price,
      stop_price=stop_price,
    )

  @function_tool
  async def list_triggers() -> Dict[str, Any]:
    """List stored auto triggers (buy/sell ideas) for follow-up."""
    return {"triggers": memory.latest_triggers()}

  @function_tool
  async def list_coins() -> Dict[str, Any]:
    """List the current active coin universe (dynamic if enabled)."""
    return {"coins": memory.get_coins(default=list(allowed_symbols))}

  @function_tool
  async def add_coin(symbol: str, reason: str) -> Dict[str, Any]:
    """Add a coin to the active universe (requires reason). The symbol is validated against KuCoin first."""
    if not symbol:
      return {"error": "symbol required"}
    norm = _normalize_symbol(symbol)
    try:
      kucoin.get_ticker(norm)
    except Exception:
      return {"error": f"Symbol {norm} not found on KuCoin. Verify the symbol exists before adding."}
    entry = memory.add_coin(norm, reason)
    return {"added": entry, "coins": memory.get_coins(default=list(allowed_symbols))}

  @function_tool
  async def remove_coin(symbol: str, reason: str, exit_plan: str) -> Dict[str, Any]:
    """Remove a coin from the active universe with an exit plan noted."""
    if not symbol:
      return {"error": "symbol required"}
    if not exit_plan:
      return {"error": "exit_plan required to remove coin"}
    entry = memory.remove_coin(_normalize_symbol(symbol), reason, exit_plan)
    return {"removed": entry, "coins": memory.get_coins(default=list(allowed_symbols))}

  @function_tool
  async def log_sentiment(symbol: str, score: float, rationale: str, source: str = "") -> Dict[str, Any]:
    """Store a sentiment score (0-1) with rationale and source for gating trades."""
    try:
      score_val = float(score)
    except (TypeError, ValueError):
      return {"error": "Invalid score"}
    if not symbol:
      return {"error": "symbol required"}
    entry = memory.log_sentiment(_normalize_symbol(symbol), score_val, rationale, source)
    return {"sentiment": entry}

  @function_tool
  async def log_decision(
    symbol: str,
    action: str,
    confidence: float,
    reason: str,
    pnl: float | None = None,
    paper: bool = False,
  ) -> Dict[str, Any]:
    """Record decision/confidence for calibration."""
    if not symbol:
      return {"error": "symbol required"}
    try:
      conf = float(confidence)
    except (TypeError, ValueError):
      return {"error": "Invalid confidence"}
    entry = memory.log_decision(_normalize_symbol(symbol), action, conf, reason, pnl=pnl, paper=paper)
    return {"decision": entry}


  # ── News & research sources ─────────────────────────────────────────────────────────────────────
  @function_tool
  async def fetch_kucoin_news(limit: int = 10) -> Dict[str, Any]:
    """Fetch latest Kucoin news RSS (lang=en). Returns list of {title, link, pubDate}."""
    url = "https://www.kucoin.com/rss/news?lang=en"
    try:
      resp = requests.get(url, timeout=10)
      resp.raise_for_status()
    except Exception as exc:
      return {"error": f"fetch_failed: {exc}"}

    try:
      import xml.etree.ElementTree as ET

      root = ET.fromstring(resp.content)
      items: List[Dict[str, Any]] = []
      for item in root.findall(".//item")[: max(1, min(int(limit or 0), 20))]:
        title_el = item.find("title")
        link_el = item.find("link")
        date_el = item.find("pubDate")
        items.append(
          {
            "title": (title_el.text or "").strip() if title_el is not None else "",
            "link": (link_el.text or "").strip() if link_el is not None else "",
            "pubDate": (date_el.text or "").strip() if date_el is not None else "",
          }
        )
      return {"items": items}
    except Exception as exc:
      return {"error": f"parse_failed: {exc}"}

  @function_tool
  async def fetch_coindesk_news(limit: int = 10) -> Dict[str, Any]:
    """Fetch latest CoinDesk news via RSS. Returns list of {title, link, pubDate, description}."""
    url = "https://www.coindesk.com/arc/outboundfeeds/rss"
    try:
      resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
      resp.raise_for_status()
    except Exception as exc:
      return {"error": f"fetch_failed: {exc}"}

    try:
      import xml.etree.ElementTree as ET

      root = ET.fromstring(resp.content)
      items: List[Dict[str, Any]] = []
      for item in root.findall(".//item")[: max(1, min(int(limit or 0), 20))]:
        title_el = item.find("title")
        link_el = item.find("link")
        date_el = item.find("pubDate")
        desc_el = item.find("description")
        items.append(
          {
            "title": (title_el.text or "").strip() if title_el is not None else "",
            "link": (link_el.text or "").strip() if link_el is not None else "",
            "pubDate": (date_el.text or "").strip() if date_el is not None else "",
            "description": (desc_el.text or "").strip() if desc_el is not None else "",
          }
        )
      return {"items": items}
    except Exception as exc:
      return {"error": f"parse_failed: {exc}"}

  @function_tool
  async def log_research(topic: str, summary: str, actions: List[str]) -> Dict[str, Any]:
    """Record a research note/strategy idea (persists in memory)."""
    if not topic or not summary:
      return {"error": "topic and summary required"}
    return memory.save_plan(title=f"Research: {topic}", summary=summary, actions=actions, author="Research Agent")

  @function_tool
  async def add_source(name: str, url: str, reason: str) -> Dict[str, Any]:
    """Add a data/research source to memory."""
    if not name or not url:
      return {"error": "name and url required"}
    entry = memory.save_plan(title=f"Source: {name}", summary=url, actions=[reason or ""], author="Research Agent")
    return {"source": entry}

  @function_tool
  async def remove_source(name: str, reason: str) -> Dict[str, Any]:
    """Mark a data/research source as removed."""
    if not name or not reason:
      return {"error": "name and reason required"}
    entry = memory.save_plan(title=f"Removed Source: {name}", summary=reason, actions=[], author="Research Agent")
    return {"removed": entry}

  return SimpleNamespace(
    place_market_order=place_market_order,
    place_limit_order=place_limit_order,
    place_spot_stop_order=place_spot_stop_order,
    cancel_spot_stop_order=cancel_spot_stop_order,
    cancel_spot_limit_order=cancel_spot_limit_order,
    list_spot_stop_orders=list_spot_stop_orders,
    set_spot_position_protection=set_spot_position_protection,
    decline_trade=decline_trade,
    fetch_recent_candles=fetch_recent_candles,
    fetch_orderbook=fetch_orderbook,
    analyze_market_context=analyze_market_context,
    plan_spot_position=plan_spot_position,
    place_futures_market_order=place_futures_market_order,
    place_futures_limit_order=place_futures_limit_order,
    place_futures_stop_order=place_futures_stop_order,
    cancel_futures_order=cancel_futures_order,
    list_futures_stop_orders=list_futures_stop_orders,
    list_futures_positions=list_futures_positions,
    get_recent_fills=get_recent_fills,
    get_closed_positions=get_closed_positions,
    fetch_funding_rate=fetch_funding_rate,
    fetch_open_interest=fetch_open_interest,
    scan_futures_market=scan_futures_market,
    fetch_futures_orderbook=fetch_futures_orderbook,
    fetch_futures_mark_price=fetch_futures_mark_price,
    fetch_futures_candles=fetch_futures_candles,
    fetch_contract_details=fetch_contract_details,
    set_futures_position_protection=set_futures_position_protection,
    transfer_funds=transfer_funds,
    fetch_account_state=fetch_account_state,
    refresh_fee_rates=refresh_fee_rates,
    save_trade_plan=save_trade_plan,
    latest_plan=latest_plan,
    latest_items=latest_items,
    clear_plans=clear_plans,
    set_auto_trigger=set_auto_trigger,
    list_triggers=list_triggers,
    list_coins=list_coins,
    add_coin=add_coin,
    remove_coin=remove_coin,
    log_sentiment=log_sentiment,
    log_decision=log_decision,
    fetch_kucoin_news=fetch_kucoin_news,
    fetch_coindesk_news=fetch_coindesk_news,
    log_research=log_research,
    add_source=add_source,
    remove_source=remove_source,
  )
