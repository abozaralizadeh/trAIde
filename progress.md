trAIde Profit-Boosting Strategy Improvements

## Work Plan (active)
1) Enhanced Signals & Filters
   - [x] Build multi-timeframe indicator analysis (EMA/RSI/MACD/ATR/Bollinger/VWAP) with context summaries the agent can call before trading.
   - [x] Update agent guidance to require indicator alignment (trend + momentum + volatility) before entries.
2) Risk Management & Position Sizing
   - [ ] Add ATR-based stop/target helpers, risk-per-trade sizing, and portfolio-level limits (daily drawdown, trades per coin/day, kill switch). (Partial: ATR sizing + per-symbol daily cap + USDT-based daily drawdown kill switch + 1h-alignment requirement + timeframe-conflict gate shipped; fill/PnL-based kill switch pending.)
   - [x] Keep 10% USDT reserve hard check and enforce maxPositionUsd per idea across spot/futures.
3) AI/LLM Usage Refinement
   - [x] Structure prompts around quant data, confidence gating, and log confidence vs. outcomes; prefer rules where possible. (Decision logging tool added; instructions updated.)
   - [x] Add optional sentiment filter hook (news/social score) to gate buys when neutral/negative. (Env flag + hard gate added.)
4) Backtesting & Validation
   - [x] Wire a lightweight backtest harness + parameter sweeps (RSI levels, stop_pct, position sizing) and compare vs. buy/hold. (CLI with sweep support added.)
   - [ ] Keep detailed trade/PnL logs for iterative improvement and regression checks. (Partial: decision/trade logs stored; PnL capture pending.)

## Execution Log
- **1h-alignment requirement + TF-conflict gate** (2026-05-28): Three more HYPE losses overnight 2026-05-27/28 after the anti-FOMO fix shipped: -$1.25 (05-27 13:38), -$1.89 (05-27 22:55), -$0.23 (05-28 03:59), cumulative HYPE loss -$5.94 over 7 entries in 36h. Root cause: anti-FOMO only blocks at `daily_exhausted=True` (RSI ≥70/≤30); after the May 26 crash HYPE's daily RSI cooled to 55-65 so the block disengaged, but `daily_bias_raw` stayed "bullish" because the 30-day uptrend's EMAs hadn't flipped. Bot kept buying "pullback longs at support" — the heuristic works in a healthy pullback during an uptrend but fails in a confirmed correction (price -10%+ off ATH, 1h bearish, lower highs forming). The label `daily_bias=bullish` didn't match the actual structural state `correction in progress`. Fix: signal-based 1h-alignment requirement — block long entries when `intraday_bias_1h == "bearish"` and mirror for shorts (futures). Added `intraday_bias_1h` to `summarize_multi_timeframe`, propagated into `_daily_gate_state`, wired `1H ALIGN BLOCK` into all 4 entry paths. Catches the bounce-entry case the prior TF-conflict logic missed (when 15m briefly turns bullish on a dead-cat bounce, 1h alignment still rejects because 1h is bearish). TF CONFLICT BLOCK kept as secondary safety net for 15m-vs-higher-TF disagreement. Rejected an earlier proposal to add a punishment-based "per-symbol consecutive loss block" — that approach blocks based on the bot's PnL history rather than market signals, which suppresses symptoms without fixing the decision logic. The 1h-alignment gate is fully signal-based: it releases as soon as 1h turns neutral/bullish (no fixed cooldown). Also tightened the `aggressive_shorts` oiPriceHint to warn against contrarian longs in confirmed downtrends instead of inviting "fade the shorts" rationalization. 112/112 tests pass.
- **Squeeze-breakout signal + volatility-scaled TP** (2026-05-27): Extended the bot from purely-defensive volatility handling to also exploiting volatility. Added `bbw_prev` to `summarize_interval` and computed a structured `squeeze_breakout` signal in `summarize_multi_timeframe` that fires only on the fresh 1h transition out of a Bollinger squeeze (BBW expanding ≥25% off the floor below 2%, ADX>20, price beyond BB band, RSI confirming). Propagated to `_daily_gate_state` and surfaced in the agent prompt with required volume-confirmation (≥1.5× 20-candle average). Anti-FOMO block still wins at exhaustion — squeezes near tops are statistically more likely to be head fakes. Added volatility-scaled `target_rr` in `plan_spot_position` — effective RR widens up to 2× base when daily ATR ≥4% (capped at daily ATR 10%), so a HYPE-like 8% ATR trade gets +67% wider TP, recovering ~93% of the dollar-PnL on winners that the quadratic ATR soft-gate trims off on entry size. Research basis: ETH D1 squeeze-breakout backtest = 44% WR, 1.59 profit factor; ATR-scaled TP performs best in trending markets and pairs naturally with the existing partial-TP staging. 112/112 tests pass.
- **Anti-FOMO at daily exhaustion** (2026-05-27): Diagnosed HYPE -2.57 USDT loss on 2026-05-26 — bot bought at $62.78 / added at $63.18 right at HYPE's $64.59 ATH (daily RSI 76), then stopped at $61.22 during the -10% same-day crash. Root cause: at daily RSI ≥70 the analytics neutralized `daily_bias`, which disabled the daily gate and let continuation longs through. Volatility soft-gate (introduced 2026-05-26 morning) only reduced size by ~3% at ATR 8.2%, vs the prior hard block at 5%. Fix: preserve `daily_bias_raw` in `analytics.py`, propagate it to `_daily_gate_state`, and add an `ANTI-FOMO BLOCK` in all four entry paths (spot market, spot limit, futures market, futures limit) plus an `ANTI-FOMO STACKING` rule in the futures anti-stacking section that blocks adds to a position when daily is exhausted in the same direction. Tightened ATR soft-gate from linear to quadratic scaling (`(threshold/ATR)²`) and lowered default `MAX_ATR_PCT_FOR_ENTRY` from 8 to 6 (HYPE-like 8.2% ATR now scales to ~54% size or hard-blocks above 9%). 112/112 tests pass.
- Added `analyze_market_context` tool to compute EMA/RSI/MACD/ATR/Bollinger/VWAP on 15m + 1h, with multi-timeframe bias/volatility summary returned to the agent.
- Strengthened agent instructions to gate entries on indicator alignment and elevated-volatility awareness; decline_trade preferred on mixed signals.
- Added `plan_spot_position` tool for ATR-based stop/target and risk-per-trade sizing with 10% reserve clipping; enforces per-symbol daily trade caps on spot and futures orders and logs trades to memory for limit checks.
- Added USDT-based daily drawdown tracking with a kill switch (configurable via `MAX_DAILY_DRAWDOWN_PCT`); supervisor loop skips agent runs once tripped.
- Added sentiment filter (env flags), sentiment logging tool, and hard gating of orders when sentiment is missing/low; added decision logging tool and instructions to log confidence.
- Added lightweight indicator backtest harness (`src/backtest.py`) for EMA/RSI/ATR stop/target strategies with CLI and simple parameter sweep output (top results).
- Added Kucoin news RSS tool (`fetch_kucoin_news`) and wired instructions to review news before scoring sentiment.

## Enhanced Strategy & Indicators
- Add momentum/trend signals (EMA, RSI, MACD) to refine entries/exits. Example with `pandas_ta`:
```python
import pandas_ta as ta

df = fetch_price_history(symbol, interval="1h", limit=100)
df["ema_fast"] = ta.ema(df["close"], length=12)
df["ema_slow"] = ta.ema(df["close"], length=26)
df["rsi"] = ta.rsi(df["close"], length=14)

if df["ema_fast"].iloc[-1] > df["ema_slow"].iloc[-1] and df["rsi"].iloc[-1] > 50:
    order = client.create_order(symbol, Side.BUY, "MARKET", funds=...)
```
- Use volatility/context filters: ATR or Bollinger Bands for dynamic stops; VWAP or pivot points to gate longs (e.g., only buy above VWAP or on pivot break).
- Refine triggers: replace single `PRICE_CHANGE_TRIGGER_PCT` with multi-factor rules (price spike + indicator confirmation) and align multiple timeframes (e.g., 1h momentum + 15m crossover).

## Risk Management & Position Sizing
- Attach stops/targets on every trade; enforce risk/reward (e.g., 1:2). Pseudo-code:
```python
entry = float(client.get_ticker(symbol)["price"])
stop = entry * 0.98
target = entry * 1.04
buy = client.create_market_order(symbol, "BUY", funds=...)
client.create_stop_order(symbol, "SELL", size=buy["filledSize"], stopPrice=stop)
client.create_limit_order(symbol, "SELL", size=buy["filledSize"], price=target)
```
- Size by risk, not fixed USD. Example ATR-based sizing:
```python
account_usd = get_total_balance_usd()
risk_per_trade = 0.01 * account_usd
atr = ta.atr(df["high"], df["low"], df["close"], length=14).iloc[-1]
stop_distance = atr * 1.5
position_usd = risk_per_trade / stop_distance
size = position_usd / entry_price
```
- Add portfolio-level limits: max daily drawdown, max trades per coin/day, and a kill switch when thresholds breach.

## AI/LLM Usage Refinement
- Use structured prompts with quant data; require crisp action + confidence:
  - `"Given [SYMBOL] moved +3% in 1h with RSI=60 and MACD positive, should the agent BUY, SELL, or HOLD? Return action and confidence (0–1)."`
- Integrate sentiment as a filter (news/social scrape → sentiment score; gate buys when score > 0.5).
- Reduce LLM overhead: favor rule-based triggers; reserve LLM for periodic analysis. Remove/simplify if it hurts PnL.
- Confidence gating: tune `MIN_CONFIDENCE` and log model confidence vs. outcomes to calibrate.

## Backtesting & Validation
- Backtest changes (Backtrader/Zipline); track win rate, profit factor, max drawdown, PnL after fees/slippage.
- Optimize parameters via grid search or walk-forward (RSI levels, stop_pct, position sizing).
- Benchmark vs. current bot and buy/hold; keep detailed trade/PnL logs for iterative improvement.

## Citations
- Best Indicators for the Most Profitable Day Trading Strategy — https://www.tradevision.io/blog/best-indicators-for-the-most-profitable-day-trading-strategy/
- RSI Indicator: Buy and Sell Signals — https://www.investopedia.com/articles/active-trading/042114/overbought-or-oversold-use-relative-strength-index-find-out.asp
- Risk Management Strategies for Algo Trading — https://www.luxalgo.com/blog/risk-management-strategies-for-algo-trading/
- Sentiment Trading with Large Language Models — https://arxiv.org/abs/2412.19245
- Algorithmic Trading Strategy Using Technical Indicators — https://www.researchgate.net/publication/371714766_Algorithmic_Trading_Strategy_Using_Technical_Indicators
