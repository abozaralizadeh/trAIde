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
| `src/regime.py` | Regime-aware entry adjustments, daily gate, trend-aligned/reversal short & long allowances |
| `src/edge.py` | Adaptive edge: per-symbol RR floor, bench/rotation, expectancy-based sizing, entry-quality stats |
| `src/config.py` | All config dataclasses + `load_config()` env loader |
| `src/memory.py` | `MemoryStore` — persists trades/decisions/events to `.agent_memory.json`; `realized_closes()` |
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
