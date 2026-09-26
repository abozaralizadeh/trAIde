# CLAUDE.md

Guidance for working in the **trAIde** repo. Read this first; it captures the design philosophy and conventions that are easy to violate.

## What this is

An autonomous crypto trading bot for KuCoin (Spot + Futures). Three LLM agents (Azure OpenAI, gpt-5.x) plus code-driven guardrails:

- **Trading Agent** (`src/agent.py`) — analyzes the market and places/manages orders each poll.
- **Research Agent** — deeper thesis work (news, web search, multi-timeframe structure) feeding the Trading Agent.
- **Supervisor Agent** (`src/supervisor.py`) — Telegram-facing oversight; can inspect state and leave notes.
- **ProtectionManager** (`src/protection.py`) — pure-code profit/risk guards enforced *outside* the LLM.
- **DashboardPublisher** (`src/dashboard_publisher.py`) — publishes a sanitized public view to Azure (Blob + Table); the separate SandBox repo renders it.

Live account is small (~$70), futures-focused, `PAPER_TRADING=false`. Fees are a real structural constraint at this size.

## Core design philosophy — DO NOT VIOLATE

This is the single most important thing and the source of most past rework:

> **Code enforces SURVIVAL. The model owns OPPORTUNITY.**

- **Survival = code's job:** risk-per-trade caps, atomic bracket orders (TP/SL attached at entry), circuit breakers, exit/trailing mechanics, correlation/concentration limits. Deterministic, tested, in `src/protection.py`, `src/regime.py`, `src/edge.py`, `src/tools.py`.
- **Opportunity = the model's job:** which coin, which direction, entry timing, when to stand aside. **Do NOT add hardcoded gates that override the model's opportunity decisions.** When the model is choosing badly, the fix is to **surface better data + reasoning ("wisdom")** and let it decide — e.g. the `entryMap`, entry-quality feedback, screener — *not* another veto. The user has pushed back hard on this repeatedly.
- **Self-tuning, no maintenance:** prefer defaults that adapt to the trade's own data (R-multiples, rolling expectancy, per-symbol bench) over magic numbers the user must hand-tune. Goal: the system gets better with better models, without config babysitting.
- **Don't overfit to a small/one-regime sample.** Recent history is often all-chop or all-trend; a value that "wins the replay" can be a trap. Pick principled values and say why in a comment.

Full context lives in the auto-memory (`memory/MEMORY.md` index) — the `project_*` notes record *why* many guards exist. Skim relevant ones before changing a guard.

## Run & test

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.main            # runs the poll loop (needs .env)
```

Production runs under systemd via Gunicorn (`src.wsgi:application`, see `setup_service.sh`). Changes require a **bot restart to deploy** — config is read at startup.

```bash
python -m pytest -q           # full suite (387+ tests, ~10s, no network — uses fakes)
python -m pytest tests/test_protection.py -q
```

Config validity: `python -c "from src.config import load_config; load_config()"`.

## Module map

| Module | Responsibility |
|---|---|
| `src/agent.py` | Trading Agent prompt + tool wiring; the LLM's decision surface |
| `src/tools.py` | Agent tools: market analysis, order placement, futures sizing, daily/regime gate enforcement (largest file) |
| `src/protection.py` | ProtectionManager + `decide_protection` (breakeven ratchet, trailing lock, early-cut, give-back) |
| `src/position_context.py` | `trade_context`: facts about the trade behind an open position (carry hold on the funding clock, noise band, original risk, restart peak) — ProtectionManager's injected lookup; `entry_thesis` (the model's per-position entry record) and `exit_probe_inputs` (replay inputs + entry-bias tags) |
| `src/regime.py` | Regime-aware entry adjustments, daily gate, trend-aligned/reversal short & long allowances |
| `src/edge.py` | Adaptive edge: per-symbol RR floor, bench/rotation, expectancy-based sizing, entry-quality stats, per-family/per-side stake (`family_stake_status`), confidence informativeness (report-only), execution map (fill rate / realized R by resting distance + every-call counterfactual, report-only) |
| `src/config.py` | All config dataclasses + `load_config()` env loader |
| `src/memory.py` | `MemoryStore` — persists trades/decisions/events to `.agent_memory.json`; `realized_closes()`; `analysis_failures` (data-quality refusals for the screener); `gate_probes` + folded `gate_state_days` (gate scoreboard, never in `signal_probes()`) |
| `src/analytics.py` | Indicators, candle validation (+ evidence-derived retry), multi-TF summary, `market_state` (breadth over the screener's universe, BTC 24h/72h/ADX — recording only) |
| `src/main.py` | Poll loop, entry-expiry handling, orchestration |
| `src/kucoin.py` | KuCoin Spot + Futures REST client |
| `src/supervisor.py` | Supervisor Agent + its inspection tools |
| `src/dashboard_publisher.py` | Sanitized Azure publish (see disclosure policy below) |

## Conventions & gotchas

- **Config changes: update BOTH the dataclass default AND the `os.getenv(...)` loader fallback in `config.py`.** A past bug had the fix in the dataclass default while the loader still returned the old value, so it never deployed. Live always goes through the loader.
- **Keep `README.md` and `.env.example` in sync in the same change** when adding/changing a feature or env var.
- **New agent-event kinds must be whitelisted in TWO places in `memory.py`** (`queue_agent_event` and the read-time sanitizer) or the event is silently dropped.
- **Risk vs leverage are separate.** Position size is stop-defined risk capped at `RISK_PER_TRADE_PCT`; leverage is only margin utilization, hard-capped (`MAX_ENTRY_LEVERAGE`), and is *not* the conviction lever — size is.
- **Exits work in R-multiples** anchored to the trade's *original* risk (via `risk_override`), so R-logic survives the stop reaching breakeven.
- **Dashboard disclosure policy:** never publish balances, equity, position sizes, or account IDs. `DASHBOARD_DISCLOSURE=normalized` = % returns + indexed curve only, no `$`. Respect it in `dashboard_publisher.py`.
- **Commit only when asked.** On a feature branch (`master` is the main branch). Tests must pass before proposing a commit.
- Tests use local fakes only — no network. Keep it that way.
- **Every agent run is its own `asyncio.run` (in a worker thread), but the OpenAI client is built ONCE in `main.py`.** Anything async that is created once and reused (httpx pools, locks, sessions) must be per-event-loop, or run #2 dies with `Event loop is closed`. See `_RetryDeploymentMissTransport` and its loopback test.
- **Probes settle on the FUTURES MARK, never the spot ticker.** The probe base is the futures mark, so its forward price must be too (`main._settle_probes_on_futures`); the spot `live_prices` map is only for auto-triggers. Settling on spot scored the perp/spot basis as `funding_carry` edge until 2026-09-25. Stored pre-fix rows are re-settled with `scripts/resettle_probes_futures.py`, which `setup_service.sh` runs automatically (bot stopped) whenever its pending-row check — which must mirror the script's own selection rules, or every deploy re-stops the bot — reads > 0 (test it on copies/fakes only — never on the live `.agent_memory.json`).
- **Funding is PER SETTLEMENT, on each contract's own clock** (KuCoin: 1h/4h/8h, shortened when funding is extreme). Never assume 8h: the carry hold uses `regime.first_settlement_after` on the exchange's `fundingTime`/`granularity` (live `main._FundingClock` → `entryContext.funding` → 8h grid only as fallback), and model-facing text prints the real interval via `regime.funding_rate_label`.
- **Restart-safe trail peak lives in PRICE space on ProtectionManager's own lifecycle identity.** `memory.peak_fe_key` (openTime, qty-derived side, |qty|, avgEntry) is shared by the writer (`update_position_extremes` → `peakFePx`) and the reader (`position_context.trade_context`); never key on KuCoin's raw `positionSide` (BOTH/LONG/SHORT) and never convert `peakPnl`/contracts (that is price × multiplier). The manager applies it only on a lifecycle's first sighting.
- **An agent close is scored against the replayed LIVE exit stack, not the bare bracket.** `protection.replay_protection_stack` (1m FUTURES bars, column order checked, ProtectionManager's EFFECTIVE `protection.cfg`) runs from `main._score_exit_probe_stacks` once a close's 8h horizon passes; `exitDiscipline` uses `stack.stackR`, keeps `bracketR` for audit, and keeps the trail's own exits (`otherExits`) on the bracket. If you change `decide_protection`'s signature or the exit-probe fields, keep the replay, the recorder (`position_context.exit_probe_inputs`) and the backfill in `scripts/resettle_probes_futures.py` in step.
- **The model sees its own entry per open position (`entryThesis`, `position_context.entry_thesis`).** Prompt text that says "fresh"/"new" about a signal must mean *changed since entry* against that block; never write a rule the model can fire on the condition it entered into. Keep one-regime numbers in code comments, never in the prompt.
- **The stand-aside is ONE stateless bar (t = net/SE < 1 → zero stake), judged per SIDE.** It is not hysteresis; don't describe it as such and don't add state to it without evidence (stateful release would have re-opened all nine 2026-09-24 refusals). `edge.family_stake_status` is the single rule (`family_stand_aside` is its boolean view); the order path passes the order's `side`, and a side with < 20 own probes falls back to the pooled row but is capped at the explore floor. Refusal text must branch on `reason` ('no edge' vs 'unproven') — never tell the model "NO EDGE" about a positive net. The scoreboard the model reads (`annotate_family_stakes`) must use the same functions, never a copy.
- **`entryContext.sizing` holds equity and $ risk — local memory only.** It is built by `tools.sizing_breakdown`, which must stay total (WARNING + None, never raise into the order). The dashboard publisher is whitelist-based; never serialise `entryContext` wholesale (a test pins this).
- **Entry distance and crossing are read from the live execution map, never from frozen figures.** `edge.execution_map` (→ `edgeReport.executionMap`, placement note via `tools.execution_bucket_note`) counts placements with `memory.is_limit_entry_record` — the SAME predicate as `limitFillRate`, so keep one predicate. Its counterfactual needs each probe's `atr15Pct`/`plannedStop`/`leaseMin` (stamped by `tools.probe_execution_stamp`) and `leaseLow`/`leaseHigh` (stamped at settle from 1m FUTURES bars via `main._PollLeaseExtremes`). The RR gate is a fee guard, not a quality filter; don't reintroduce the Jul-30 crossing/TTL numbers (+13.05R, -0.37R, +0.388R) — they did not reproduce.
- **Reachability is ONE check: `agent._bot_can_trade` over the cached live spot list (`tools._SPOT_SYMBOLS`).** The screener (`tradeable=` → `excluded`, applied before the top-n cut) and `add_coin` both call `tools._tradeable_by_bot`; never add a second copy. `None` (list unavailable) means fail OPEN. Perp-only contracts stay unreachable on purpose (stock perps gap across their session close). Futures tools take the model's `X-USDT`: read-only data tools resolve outside the universe via the live contract list (`_resolve_futures_symbol`), order tools do not, and `list_futures_stop_orders` calls KuCoin unfiltered and filters client-side (KuCoin rejects `X-USDT` with 100003).
- **`marketState` is recording only.** `main._MarketStateClock.refresh()` is the ONLY network path (poll loop, ≤ hourly); the order path, the agent and the exit recorder read `.current()`. Splits key on rolling breadth24 terciles (`edge.breadth_terciles`), never fixed cut-offs. Never show the model a per-state R line (`agent._exit_discipline_for_prompt` strips `trailByMarketState`; the agent's `signal_edge_stats` runs without `market_state_split`) — the only evidence is one rally.
- **Every `{"rejected": True}` in `_place_futures_limit_order_impl` before the signal probe carries a `gate` code** from `memory.SCORED_GATES` (the tool wrapper records a gate probe) or `memory.STRUCTURAL_REFUSALS` (records nothing) — an AST test fails otherwise, so a new gate must say which it is. `tools.directional_gates_against` is a second statement of the gate conditions (a state reading has no call to run them on); a parametrised test pins its first element to the gate the order path returns — change both together. The gate scoreboard (`edge.gate_scoreboard`) is REPORT-ONLY: dashboard, Supervisor, hourly log — never the trading prompt or `agent.py` (a test greps for it), and never an input to a gate.
- **Measurement I/O on the poll thread is bounded; survival reads caches only (2026-09-25 review).** `ProtectionManager.run` runs under `order_lock` and must never make a network call: `_FundingClock` lookups are cache-only and the loop calls `refresh(carry_refresh_targets(...))` before the pass, OUTSIDE the lock. Probe settlement / stack replays / market state share one `_MeasurementBudget` (a quarter of the poll interval), ask the most urgent rows first, back off per symbol (`_MeasureBackoff`) and use the 5s `with_timeout` client — a new measurement fetch must do the same. Report-only telemetry must never call a caching decision helper (`_btc_daily_bias` froze the live correlation veto) — peek at caches instead.
- **Family verdicts are scored over each family's HOLDING-TIME MIX, never a snapped single horizon** (`edge.family_horizon_weights` → `signal_edge_stats(family_horizon_weights=...)` → `_mixed_row`). Snapping the median hold made the verdict a step function of one close (Sep 25: continuation 240m → 60m on a fast FET winner, +1.1% → −0.15%, benched mid-rally). Every verdict path (agent state — which the ORDER PATH reads —, dashboard, Supervisor, resettle report) passes the weights; a test pins all four. `family_scoring_horizons` (snapped) remains only for report-only consumers.
- **A stood-aside side re-opens ONLY through new calls on it.** Never write prompt or refusal text that tells the model to stop proposing a benched side ("stand down until the regime turns", "don't spend a turn") — a decline records nothing, and that froze the verdict on Sep 25-26 exactly like the Sep 4-7 probe starvation. The prompt asks the model to SUBMIT genuine benched setups (refusal expected, call recorded).
- **A repeat of the same call (symbol, side, family) inside the family's SHORTEST scored horizon is not stored** (`memory.record_signal_probe(min_gap_sec=...)`, gap from `tools.repeat_gap_seconds` over `family_horizon_weights`). It can never be a new de-overlapped observation, but the per-family cap keeps the NEWEST 150 rows, so repeats would evict independent evidence until a verdict "un-learns" itself (Sep 26: continuation at the cap with 88 independent rows while the model re-submitted the same benched SOL long every run). The refusal tells the model the repeat added nothing.
- **Spot dust is not a position** (`utils.SPOT_DUST_VALUE_USD`, used by `agent.reconcile_spot_positions`, `main._discover_unlisted_holdings` and `tools.remove_coin`): a $0.005 KCS remainder once sent the model to place TP/SL on dust every few runs.
- **Stakes are per SIDE; unknown is not zero.** A pooled `by_family` row carries `stakeBySide`/`standAsideBySide` and is `standAside` only when both sides are (no pooled scalar stake anywhere — the order path always passes the side). A failed funding lookup is `f{h}` None (backfilled, skipped in carry scoring), never absent-as-zero. `exitDiscipline`'s verdict uses stack-scored closes only; bracket-only rows are audit.
- **`story.md` is the running build journal** (timeline, numbers, themes, screenshot checklist) and the raw material for posts like `medium_post.md`. Add a dated line when something story-worthy happens — a bug with a good number, a reversal, a milestone.
