# Profitability Improvement Plan for trAIde

## Context

The trAIde bot is a 3-agent crypto trading system (Trading, Research, Supervisor) running on KuCoin with Azure OpenAI (gpt-5.2). It already has solid foundations: 11+ technical indicators, market regime detection (trending/ranging/squeeze via ADX+BBW), multi-timeframe analysis (15m+1h), ATR-based position sizing, futures with 3x leverage, and fee-aware TP/SL. Recent supervisor notes improved profitability by shifting to futures focus, 10% risk per trade, fee-aware TP, and mandatory research calls.

The goal is to identify and prioritize **concrete, implementable changes** that will increase net profitability — ranked by effort-to-impact ratio.

---

## TIER 1: HIGH IMPACT, LOW EFFORT

### 1. Staged Take-Profit (Partial Exits)

**Problem:** The bot uses a single TP target. When price reaches 80% of TP then reverses, the entire position closes at a loss or breakeven. The `performanceSummary` already tracks `missedProfit` — positions where `peakPnl > 0` but closed lower — confirming this is a real issue.

**Solution:** Scale out of positions in 2-3 tranches instead of one all-or-nothing TP.

**Implementation:**
- Modify `place_market_order()` and `place_futures_market_order()` in `agent.py` to place multiple TP orders at different levels
- Tranche 1 (40% of position): TP at 1.0x ATR profit — lock in base profit
- Tranche 2 (30%): TP at 1.5x ATR — capture extended move
- Tranche 3 (30%): Trailing stop at 1.5x ATR below peak — ride runners
- For **range trades** (regime="ranging"): 2 tranches — 60% at BB midline, 40% with tighter trail (0.75 ATR)
- Add `partial_tp_enabled` config flag to `config.py`
- Update `set_spot_position_protection()` and `set_futures_position_protection()` tools to accept multiple TP levels
- Modify the agent system prompt in `agent.py` to instruct the LLM to use staged exits

**Files:** `agent.py`, `config.py`

**Expected impact:** Capture 20-40% more of profitable moves. Directly addresses the `missedProfit` metric already being tracked.

---

### 2. Add 4H Timeframe as Directional Filter

**Problem:** The bot only uses 15m + 1h. Without a higher timeframe filter, the agent frequently enters trades against the dominant trend, leading to stopped-out positions.

**Solution:** Add 4-hour candles as a directional bias filter. Only enter in the direction of the 4H trend unless in a confirmed ranging regime.

**Implementation:**
- In `analytics.py`, add `"4hour": 14400` to `INTERVAL_SECONDS` dict (line 9)
- In the `analyze_market_context()` tool in `agent.py`, fetch 4H candles alongside 15m and 1h
- Modify `summarize_multi_timeframe()` in `analytics.py` to support 3 timeframes with weighted scoring:
  - 4H: 40% weight (directional filter)
  - 1H: 35% weight (setup identification)
  - 15m: 25% weight (entry timing)
- Add a **conflict filter**: Require 4H and 1H to agree on direction before entering trend trades. If 4H disagrees, only allow range trades or reduced-size entries
- Update agent system prompt to reference the 3-timeframe hierarchy

**Files:** `analytics.py`, `agent.py`

**Expected impact:** Filter out 30-50% of false signals by eliminating counter-trend entries. QuantPedia backtests show multi-timeframe filters on BTC achieved Sharpe 1.07 with max drawdown ~14%.

---

### 3. OI-Price Divergence Matrix (Sharpen Existing Futures Signals)

**Problem:** The bot already fetches OI and funding rate, but these are passed as raw data for the LLM to interpret. The LLM doesn't always correctly synthesize OI + price into a positioning signal.

**Solution:** Pre-compute the OI-Price matrix and pass a clear signal label to the agent.

**Implementation:**
- In `analyze_market_context()` in `agent.py`, after fetching OI and mark price, compute:
  ```
  Price Up + OI Up = "strong_trend" (stay in / add to position)
  Price Up + OI Down = "short_covering" (exit longs, don't enter new)
  Price Down + OI Up = "aggressive_shorts" (stay short / enter short)
  Price Down + OI Down = "long_capitulation" (potential reversal zone — contrarian long)
  ```
- Compute price direction from last 4H candle close vs. 4H candle open
- Compute OI direction from current OI vs. 24h ago (available from contract detail `turnoverOf24h`)
- Add `oi_price_signal` field to the market context dictionary returned to the agent
- Also add **funding rate divergence**: price rising + funding falling = "hidden_strength"; price falling + funding rising = "hidden_weakness"
- Update agent system prompt to explicitly reference these signals in entry/exit decisions

**Files:** `agent.py`

**Expected impact:** Significantly improves futures trade quality by converting raw data into actionable positioning signals. The October 2025 crash ($9.89B liquidated) was a textbook OI divergence.

---

### 4. Circuit Breakers & Portfolio Heat Limits

**Problem:** The bot tracks drawdown but has no kill switch — the agent "decides how to adapt." During rapid cascading losses, LLM reasoning can be slow or make poor decisions under pressure.

**Solution:** Add code-level (not LLM-level) circuit breakers.

**Implementation:**
- In `main.py`, add hard checks before calling `run_trading_agent()`:
  - **Daily loss limit**: If `drawdown_pct > 8%`, set `snapshot.trading_restricted = True` — agent can only close positions, not open new ones
  - **Consecutive loss limit**: Track in memory — 4 consecutive losing trades triggers a 2-hour cooldown
  - **Portfolio heat**: Total capital at risk (sum of all open position stop distances) must stay below 20% of total equity. If exceeded, block new entries
- Add `CircuitBreakerConfig` to `config.py` with `max_daily_drawdown_pct`, `max_consecutive_losses`, `max_portfolio_heat_pct`, `cooldown_minutes`
- In the snapshot JSON, add `trading_restricted` boolean and `restriction_reason` string
- Update agent system prompt to respect `trading_restricted` — close-only mode

**Files:** `main.py`, `config.py`, `agent.py`, `memory.py`

**Expected impact:** Prevents catastrophic drawdowns. Recovery from 50% drawdown requires 100% gain — prevention is exponentially more valuable. A 73% failure rate for crypto bots within 6 months is mostly due to missing circuit breakers.

---

## TIER 2: HIGH IMPACT, MEDIUM EFFORT

### 5. Fractional Kelly Criterion Position Sizing

**Problem:** Current sizing is fixed at `risk_per_trade_pct` (10% from supervisor notes). This doesn't adapt to edge quality — a high-confidence trade with 70% win rate gets the same size as a marginal trade at 60%.

**Solution:** Use quarter-Kelly sizing that dynamically adjusts based on rolling win rate and reward-to-risk ratio.

**Implementation:**
- In `memory.py`, add `kelly_fraction()` method:
  ```python
  def kelly_fraction(self, venue=None, lookback=50):
      summary = self.performance_summary(venue=venue)
      W = summary["win_rate"]
      avg_win = summary["avg_win"]
      avg_loss = abs(summary["avg_loss"]) or 0.01
      R = avg_win / avg_loss
      kelly = W - (1 - W) / R
      return max(0.01, min(0.25, kelly * 0.25))  # quarter-Kelly, floor 1%, cap 25%
  ```
- In `plan_spot_position()` and futures sizing in `agent.py`, replace fixed `risk_pct` with `memory.kelly_fraction()` when confidence > 0.7
- Multiply by **inverse volatility ratio**: `(avg_atr_pct_30d / current_atr_pct)` to auto-shrink in volatile periods
- Add config flag `kelly_sizing_enabled` and `kelly_min_trades` (minimum sample size, default 30)

**Files:** `memory.py`, `agent.py`, `config.py`

**Expected impact:** Sizes up when edge is strong, sizes down when edge is weak. Half-Kelly reduces portfolio volatility by ~25% while sacrificing only ~25% of long-term growth.

---

### 6. Volume Profile Levels (POC / Value Area)

**Problem:** The bot sets TP at ATR multiples or BB midline. These are decent but miss the structural support/resistance that institutional traders watch — Volume Profile levels.

**Solution:** Compute POC (Point of Control), VAH (Value Area High), VAL (Value Area Low) from recent candles and use them for entry/exit targeting.

**Implementation:**
- In `analytics.py`, add `compute_volume_profile()` function:
  ```python
  def compute_volume_profile(df, num_bins=50):
      price_range = df["high"].max() - df["low"].min()
      bin_size = price_range / num_bins
      bins = {}
      for _, row in df.iterrows():
          for p in np.arange(row["low"], row["high"], bin_size):
              bin_key = round(p / bin_size) * bin_size
              bins[bin_key] = bins.get(bin_key, 0) + row["volume"] * bin_size / (row["high"] - row["low"] + 1e-10)
      poc = max(bins, key=bins.get)
      # Value area: 70% of volume centered on POC
      sorted_bins = sorted(bins.items(), key=lambda x: x[1], reverse=True)
      total_vol = sum(v for _, v in sorted_bins)
      cum = 0
      va_prices = []
      for price, vol in sorted_bins:
          cum += vol
          va_prices.append(price)
          if cum >= total_vol * 0.70:
              break
      return {"poc": poc, "vah": max(va_prices), "val": min(va_prices)}
  ```
- Add to `summarize_interval()` output
- Update agent prompt: Use POC as mean-reversion anchor. Set TP at next HVN. Enter longs near VAL, shorts near VAH

**Files:** `analytics.py`, `agent.py`

**Expected impact:** Provides structurally meaningful support/resistance levels that ATR and Bollinger Bands cannot capture. Volume profile is standard on institutional desks and particularly effective in crypto where volume concentrates at psychological levels.

---

### 7. Limit Order Default (Maker Fees)

**Problem:** The bot defaults to market orders (taker fees ~0.1% spot, ~0.06% futures). With 10% risk sizing and frequent trading, round-trip fees eat 0.12-0.2% per trade.

**Solution:** Default to limit orders placed at the current bid (for buys) or ask (for sells), with a fallback to market if not filled within a timeout.

**Implementation:**
- In `place_market_order()` in `agent.py`, add a `prefer_limit` parameter (default True)
- When `prefer_limit=True`:
  1. Place a limit order at `best_bid + 1 tick` (buys) or `best_ask - 1 tick` (sells)
  2. Wait 15-30 seconds (configurable `limit_order_timeout_sec`)
  3. If not filled, cancel and place market order
- For futures: similar logic in `place_futures_market_order()`
- Savings: Spot maker fee ~0.1% vs taker 0.1% (may get KCS discount), Futures maker ~0.02% vs taker 0.06% = **0.04% saved per leg, 0.08% round-trip on futures**
- Add `prefer_limit_orders` config flag

**Files:** `agent.py`, `config.py`

**Expected impact:** Saves 0.04-0.08% per round-trip on futures. Over 100 trades/month, that's 4-8% of notional saved in fees alone.

---

### 8. Post-Trade Cooldown Timer

**Problem:** After a stop-loss hit, the bot can immediately re-enter the same trade on the next poll (30s later). This leads to revenge trading — entering the same losing setup repeatedly.

**Solution:** Enforce a minimum cooldown per symbol after a stop-loss.

**Implementation:**
- In `memory.py`, track `last_loss_ts` per symbol in decisions
- In `place_market_order()` and `place_futures_market_order()` in `agent.py`, check: if the last closed trade on this symbol was a loss and occurred within `cooldown_minutes` (default 30), reject the entry
- Add `post_loss_cooldown_minutes` to `config.py`
- Make this configurable via supervisor permanent notes

**Files:** `memory.py`, `agent.py`, `config.py`

**Expected impact:** Prevents the most common form of overtrading — revenge entries after a stop. Small impact per trade but compounds over time.

---

## TIER 3: MEDIUM IMPACT, MEDIUM-HIGH EFFORT

### 9. ATR Trailing Stop (Dynamic Stop Management)

**Problem:** Once TP/SL are placed, they're static. If price moves 80% toward TP then reverses, the original SL is hit for a full loss instead of breakeven or small profit.

**Solution:** Implement a trailing stop that moves SL to breakeven after 1x ATR profit, then trails at 1.5x ATR.

**Implementation:**
- This requires a background monitor since KuCoin doesn't natively support trailing stops
- In `main.py`, add a `_check_trailing_stops()` function called every poll cycle:
  1. For each open position, compare current price to entry price
  2. If unrealized profit > 1x ATR: move SL to breakeven (cancel old stop, place new)
  3. If unrealized profit > 2x ATR: trail SL at entry + 1x ATR
  4. If unrealized profit > 3x ATR: trail SL at peak - 1.5x ATR
- Store trail state in memory: `{symbol, entry, current_sl, trail_active, peak_price}`
- Use `cancel_spot_stop_order()` + `place_spot_stop_order()` to update SL on exchange
- Add `trailing_stop_enabled` config flag

**Files:** `main.py`, `agent.py`, `memory.py`, `config.py`

**Expected impact:** Converts many full-loss trades into breakeven or small wins. Combined with staged TP (#1), this is the most impactful exit management upgrade.

---

### 10. Correlation-Aware Position Limits

**Problem:** If the bot holds BTC-USDT long, ETH-USDT long, and SOL-USDT long simultaneously, it effectively has 3x the intended exposure to a single risk factor (crypto market direction). A single downturn hits all positions.

**Solution:** Track cross-position correlation and limit total directional exposure.

**Implementation:**
- In `analytics.py`, add `compute_correlation(candles_a, candles_b, window=20)` using rolling Pearson correlation of returns
- In `place_market_order()` / `place_futures_market_order()`, before opening a new position:
  1. Fetch candles for the new symbol + all existing open position symbols
  2. If correlation > 0.7 with any existing position on the same side, halve the new position size
  3. If total correlated exposure (sum of correlated position sizes) > `max_correlated_exposure_pct` (default 30%), reject entry
- Store correlation cache in memory (refresh daily)

**Files:** `analytics.py`, `agent.py`, `memory.py`

**Expected impact:** Prevents portfolio blow-ups from correlated positions. Crypto correlations reach 60-90% during downturns — this is a critical risk limiter.

---

### 11. Macro Event Pause

**Problem:** During FOMC, CPI releases, or exchange maintenance, volatility spikes cause random stop-outs. The bot has no awareness of scheduled macro events.

**Solution:** Fetch an economic calendar and pause new entries around high-impact events.

**Implementation:**
- Add a `fetch_economic_calendar()` tool in `agent.py` that calls a free API (e.g., ForexFactory RSS, or use web_search to check upcoming events)
- Before each agent run in `main.py`, check if a high-impact event is within 30 minutes
- If so, set `snapshot.macro_event_pause = True` — agent can only manage existing positions, not enter new ones
- Research Agent should also flag upcoming events in its research cycle

**Files:** `agent.py`, `main.py`

**Expected impact:** Avoids the worst whipsaw trades. CPI and FOMC cause 2-5% moves in seconds — no edge in guessing direction.

---

### 12. Enhanced Backtesting with Walk-Forward Validation

**Problem:** The current `backtest.py` uses in-sample optimization only. Parameters that look great on historical data may be overfit.

**Solution:** Add walk-forward analysis (WFA) and Monte Carlo simulation.

**Implementation:**
- In `backtest.py`, add `walk_forward_test()`:
  1. Split data into rolling windows: 70% train, 30% test
  2. Optimize parameters on train window
  3. Test on out-of-sample window
  4. Slide forward and repeat
  5. Compute Walk-Forward Efficiency: `WFE = OOS_profit / IS_profit` (target > 0.5)
- Add `monte_carlo_sim(trades, n_iterations=10000)`:
  1. Shuffle trade order randomly
  2. Apply random price variation (+-0.1% per trade)
  3. Compute max drawdown distribution
  4. Report 95th percentile worst-case drawdown
- Add `parameter_sensitivity()`: vary each parameter by +-20% and report performance delta. Flag parameters where small changes cause large performance swings

**Files:** `backtest.py`

**Expected impact:** Prevents deploying overfit strategies. Not a profit generator but a critical profit protector. An estimated 90% of crypto strategies are overfit.

---

## TIER 4: LONGER-TERM / EXPLORATORY

### 13. ML Meta-Strategy Selector

Use XGBoost or a simple ensemble to predict which strategy (trend-following vs. mean-reversion) will perform better in the next N hours, replacing the pure ADX+BBW regime classifier.

### 14. On-Chain Flow Signals

Integrate CryptoQuant API for exchange inflow/outflow data and whale accumulation signals as confirmation filters.

### 15. TWAP Execution for Large Orders

For entries/exits above $500 notional, split into 3-5 equal slices over 5-10 minutes to reduce market impact.

---

## Implementation Priority (Recommended Order)

| # | Change | Impact | Effort | Dependencies |
|---|--------|--------|--------|-------------|
| 1 | Circuit breakers (#4) | Loss prevention | Low | None |
| 2 | 4H timeframe filter (#2) | Signal quality | Low | analytics.py only |
| 3 | OI-Price divergence (#3) | Futures accuracy | Low | Already has data |
| 4 | Post-trade cooldown (#8) | Overtrading prevention | Low | memory.py |
| 5 | Staged take-profit (#1) | Capture more profit | Medium | TP/SL order logic |
| 6 | Limit order default (#7) | Fee savings | Medium | Order execution |
| 7 | ATR trailing stop (#9) | Exit management | Medium | Needs poll-cycle monitor |
| 8 | Kelly sizing (#5) | Adaptive sizing | Medium | performance_summary() |
| 9 | Volume profile (#6) | Better levels | Medium | analytics.py |
| 10 | Correlation limits (#10) | Risk management | Medium | analytics.py + memory |
| 11 | Walk-forward backtest (#12) | Strategy validation | Medium | backtest.py |
| 12 | Macro event pause (#11) | Avoid whipsaws | Medium | Calendar API |

---

## Verification Plan

After implementing each change:
1. **Paper trade** with `PAPER_TRADING=true` for at least 48 hours
2. **Compare metrics** via `performance_summary()`: win rate, avg win/loss, missed profit count, total PnL
3. **Backtest** the specific change using `run_backtest()` with walk-forward validation
4. **Monitor Telegram** notifications for unexpected behavior
5. **Check logs** via supervisor for rejected trades, circuit breaker activations
6. **A/B test** where possible: run old config on one coin, new config on another, compare after 1 week

---

## Quick Wins (Config Changes Only, No Code)

These can be done immediately via supervisor permanent notes or `.env`:

1. **Reduce `MAX_TRADES_PER_SYMBOL_PER_DAY`** from 100 to 5-10 — 100 is essentially unlimited and allows overtrading
2. **Increase `MIN_NET_PROFIT_USD`** from $0.05 to $0.50+ — filters out noise trades that barely cover fees
3. **Increase `MIN_PROFIT_ROI_PCT`** from 0.1% to 0.3% — ensures each trade is meaningful after fees
4. **Set `PRICE_CHANGE_TRIGGER_PCT`** from 0.3% to 0.5% — reduces noise triggers that waste LLM calls
5. **Reduce `MAX_IDLE_POLLS`** from 3 to 5-10 — fewer forced runs in quiet markets means fewer low-quality trades
