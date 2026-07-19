import os
from dataclasses import dataclass, field
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()


def _as_bool(value: str | None, fallback: bool = False) -> bool:
  if value is None:
    return fallback
  return value.lower() == "true"


@dataclass
class AzureConfig:
  endpoint: str
  deployment: str
  api_version: str
  api_key: str


@dataclass
class ApimConfig:
  endpoint: str
  deployment: str
  api_version: str
  subscription_key: Optional[str]


@dataclass
class KucoinConfig:
  api_key: str
  secret: str
  passphrase: str
  base_url: str


@dataclass
class KucoinFuturesConfig:
  enabled: bool
  base_url: str
  # "cross" | "isolated" | "auto". The bot's set_leverage uses the cross endpoint, so the
  # cross-leverage call is only issued in cross mode (avoids "switch to cross margin" errors).
  margin_mode: str = "cross"


@dataclass
class TradingConfig:
  coins: List[str]
  flexible_coins_enabled: bool
  paper_trading: bool
  max_position_usd: float
  risk_per_trade_pct: float
  min_confidence: float
  sentiment_filter_enabled: bool
  sentiment_min_score: float
  poll_interval_sec: float
  price_change_trigger_pct: float
  max_idle_polls: int
  max_leverage: float
  max_trades_per_symbol_per_day: int
  min_net_profit_usd: float
  min_profit_roi_pct: float
  estimated_slippage_pct: float
  range_trading_enabled: bool
  partial_tp_enabled: bool
  kelly_sizing_enabled: bool
  kelly_min_trades: int
  prefer_limit_orders: bool
  limit_order_timeout_sec: float
  post_loss_cooldown_minutes: float
  max_entry_leverage: float
  min_trade_interval_minutes: float
  max_24h_volatility_pct: float
  max_atr_pct_for_entry: float
  entry_limit_expiry_minutes: float
  min_entry_deviation_pct: float
  research_handoff_after_no_trade_runs: int  # force a Research handoff after N no-trade runs (0=off)
  # Risk guardrails (added after the RE-USDT concentration blowup, 2026-06-21):
  max_position_equity_pct: float = 0.5        # cap a single position's notional at this fraction of total equity (0=off)
  min_futures_listing_age_days: float = 7.0   # block entries on futures contracts younger than this (0=off)
  research_handoff_cooldown_min: float = 120.0 # min minutes between forced Research handoffs (0=off)
  min_futures_rr: float = 1.5                 # reject futures entries whose post-cost reward:risk is below this (0=off)
  screener_min_turnover_usd_24h: float = 5_000_000.0  # market screener liquidity floor (24h USDT turnover)
  atomic_bracket_enabled: bool = True         # attach TP/SL to the entry order (KuCoin st-orders) so a limit
                                              # fill is protected instantly, not on the next agent run
  emergency_sl_pct: float = 0.02              # poll-loop safety net: SL distance for an unprotected open
                                              # position (fraction of entry) until the agent sets a real one (0=off)
  # Model invocation budget: protection remains code-driven every poll, while expensive discretionary
  # analysis runs less often when flat and more often when capital is exposed.
  # When FLAT and quiet, this is the initial model HUNT cadence. Repeated no-action runs back it off.
  # Pending atomic brackets are managed by deterministic expiry and do not count as exposed capital.
  flat_agent_cooldown_sec: float = 3600.0
  active_agent_cooldown_sec: float = 300.0
  flat_backoff_max_multiplier: float = 4.0  # power-of-two no-action backoff cap
  # Adaptive price-trigger tuning (the model-call gate). A symbol's per-poll noise EWMA × the noise
  # multiplier sets its trigger threshold, capped at price_change_trigger_pct × the ceiling multiplier.
  # The ceiling bounds worst-case blindness: at 2.0 (default) any move >= 2× the base trigger always
  # gets a fresh model look, even in the noisiest symbol — biased toward safety over token savings.
  price_noise_multiplier: float = 4.0
  price_trigger_max_multiplier: float = 2.0


@dataclass
class EdgeConfig:
  """Adaptive edge controller (src/edge.py) — risk posture derived from rolling realized results.

  Self-tuning by design: when recent evidence is net-losing the bot reduces capital at
  risk, benches symbols that keep losing, and shrinks size during loss streaks — then
  relaxes back automatically once realized expectancy turns positive. Legacy adaptive-RR
  fields remain loadable for backward compatibility, but are not used by live admission.
  """
  enabled: bool = True
  lookback_trades: int = 30        # rolling window of realized closes the stats are computed over
  min_trades: int = 8              # minimum closes before adaptive actions kick in (else static behavior)
  rr_step: float = 0.5             # added to the futures RR floor while expectancy is negative
  rr_cap: float = 2.5              # ceiling for the adaptive RR floor
  rr_stale_hours: float = 18.0     # if no realized close in this long, the raise decays back to base
                                   # (a losing streak old enough to have frozen trading shouldn't keep
                                   # the bar raised — that lockout prevents the wins that would lower it)
  symbol_rr_min_trades: int = 2    # min closes ON A SYMBOL before its own RR floor is raised (per-symbol)
  bench_lookback: int = 5          # per-symbol recent closes examined for the bench
  bench_min_losses: int = 3        # losses within that lookback (with negative net) that bench the symbol
  bench_cooldown_hours: float = 12.0  # base bench rest, scaled by loss count (see bench_cooldown_max_mult)
  bench_cooldown_max_mult: int = 4    # max multiplier on the bench rest for a persistently-losing symbol
  streak_threshold: int = 2        # consecutive realized losses that trigger the size throttle
  streak_size_factor: float = 0.5  # entry-size multiplier while on a losing streak
  direction_min_trades: int = 5    # evidence required before long/short-specific adaptation
  negative_expectancy_size_factor: float = 0.5  # explore smaller on a losing direction; auto-restores


@dataclass
class CircuitBreakerConfig:
  max_daily_drawdown_pct: float
  max_consecutive_losses: int
  max_portfolio_heat_pct: float
  cooldown_minutes: float


@dataclass
class ProfitProtectionConfig:
  """Code-driven profit guards enforced outside the LLM (see src/protection.py)."""
  enabled: bool
  dry_run: bool
  breakeven_trigger_r: float
  breakeven_fee_pct: float
  giveback_pct: float
  min_favorable_excursion_pct: float
  no_chase_enabled: bool
  post_win_cooldown_minutes: float
  no_chase_buffer_pct: float
  # Give-back arming: don't let the give-back cap act until the run has reached this multiple of
  # the trade's own initial risk (stop distance). Stops the cap from strangling winners at
  # fee-scale (+0.3-0.5% ROE closes) while losses ride to the full stop. 0 = pct-arming only.
  giveback_arm_r: float = 1.0
  # Early invalidation: a trade that NEVER went meaningfully green and is failing toward its stop is
  # cut early instead of riding to the full SL. The bot's MFE/MAE data shows winners work almost
  # immediately (stay green from entry) while losers go against from the start — so this surgically
  # shrinks losers without touching winners. Cuts avg loss, the side that's been > avg win.
  early_cut_enabled: bool = True
  early_cut_grace_min: float = 20.0        # give a fresh entry this long to work before it can be cut
  early_cut_min_favorable_pct: float = 0.003  # if peak excursion never reached this (frac of entry), it "never worked"
  early_cut_mae_frac: float = 0.6          # ...and it's this far toward the stop → cut the remaining distance


@dataclass
class RegimeConfig:
  """Regime-aware entry throttle + trend-aligned shorts (see src/regime.py)."""
  throttle_enabled: bool            # raise confidence bar + shrink size in a hostile regime
  caution_min_confidence: float     # elevated confidence floor when daily is bearish/exhausted
  caution_size_factor: float        # position-size multiplier in a hostile regime
  trend_shorts_enabled: bool        # permit trend-aligned shorts past the anti-FOMO gate
  trend_short_min_confidence: float # confidence bar specifically for a counter-bounce short
  trend_short_require_15m: bool     # require 15m (not just 1h) bearish confirmation for the short
  # Correlation gate: block LONGs on non-major alts while BTC's daily regime is bearish (alts are
  # high-beta to BTC; longing them into a BTC downtrend is what blew up on RE-USDT, 2026-06-21).
  alt_long_block_enabled: bool = True
  alt_majors: tuple = ("BTC", "ETH")  # symbols exempt from the alt-long gate (they have their own daily gate)
  # A blanket BTC/alt veto misses genuine relative-strength leaders.  Permit only an alt whose own
  # daily, 4h, 1h and 15m trends are all bullish at high confidence, then size it down.  This is a
  # signal-based exception, not a symbol allow-list, so it adapts as leadership rotates.
  relative_strength_longs_enabled: bool = True
  relative_strength_min_confidence: float = 0.82
  relative_strength_size_factor: float = 0.5
  # Conviction sizing: scale position size by how far confidence clears the floor, so low-conviction
  # entries size down instead of taking full size (the SOL drawdown was full-size low-conviction shorts).
  conviction_sizing_enabled: bool = True
  conviction_full_confidence: float = 0.85   # confidence at/above which full size is used
  conviction_min_size_factor: float = 0.5    # size multiplier at the confidence floor
  # Deadlock break: in a clean daily trend, allow the daily-aligned entry past the 1h gate when a 1h
  # counter-bounce is stalling (15m no longer confirms it) — fixes the both-directions-blocked stall.
  deadlock_break_enabled: bool = True
  deadlock_min_confidence: float = 0.72      # raised confidence bar to take the trend-continuation entry
  # Reversal long: allow a LONG past a bearish daily gate when 1h+15m confirm a turn and confidence is
  # high — the lagging daily gate otherwise forbids catching reversals (missed an +11% ETH bounce).
  reversal_longs_enabled: bool = True
  reversal_long_min_confidence: float = 0.80  # high bar — counter-daily longs are only for confirmed turns
  reversal_long_require_15m: bool = True      # require 15m (not just 1h) bullish confirmation
  # Mirror: allow a SHORT past a bullish daily gate when 1h+15m confirm a roll-over (regimes flip both ways).
  reversal_shorts_enabled: bool = True
  reversal_short_min_confidence: float = 0.80
  reversal_short_require_15m: bool = True


@dataclass
class TelegramConfig:
  enabled: bool
  bot_token: str
  chat_id: str
  silent: bool


@dataclass
class SupervisorConfig:
  enabled: bool
  log_file: str
  log_max_bytes: int
  log_backup_count: int


@dataclass
class LangsmithConfig:
  enabled: bool
  api_key: Optional[str]
  project: Optional[str]
  api_url: Optional[str]
  tracing: bool
  sample_rate: float = 0.1  # head-sampling fraction of runs sent to LangSmith (avoids monthly trace limit)


@dataclass
class DashboardConfig:
  enabled: bool
  connection_string: str
  table_name: str
  container_name: str
  publish_interval_sec: float
  disclosure: str          # "normalized" (default) | "absolute" | "both"
  feed_limit: int
  index_base: float        # equity index start (default 100.0)


@dataclass
class AppConfig:
  azure: AzureConfig
  apim: ApimConfig
  kucoin: KucoinConfig
  kucoin_futures: KucoinFuturesConfig
  trading: TradingConfig
  circuit_breaker: CircuitBreakerConfig
  profit_protection: ProfitProtectionConfig
  regime: RegimeConfig
  langsmith: LangsmithConfig
  telegram: TelegramConfig
  supervisor: SupervisorConfig
  dashboard: DashboardConfig
  tracing_enabled: bool
  console_tracing: bool
  openai_trace_api_key: Optional[str]
  memory_file: str
  retention_days: int
  agent_max_turns: int
  # Adaptive edge controller — defaulted so existing AppConfig constructions keep working;
  # load_config wires env overrides explicitly.
  edge: EdgeConfig = field(default_factory=EdgeConfig)


def load_config() -> AppConfig:
  coins_env = os.getenv("COINS", "")
  coins = [c.strip() for c in coins_env.split(",") if c.strip()]
  kucoin_base_url = os.getenv("KUCOIN_BASE_URL", "https://api.kucoin.com")
  kucoin_futures_base_url = os.getenv("KUCOIN_FUTURES_BASE_URL", "https://api-futures.kucoin.com")

  config = AppConfig(
    azure=AzureConfig(
      endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
      deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5.2"),
      api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-01-preview"),
      api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
    ),
    apim=ApimConfig(
      endpoint=os.getenv("AZURE_APIM_OPENAI_ENDPOINT", ""),
      deployment=os.getenv("AZURE_APIM_OPENAI_DEPLOYMENT", ""),
      api_version=os.getenv("AZURE_APIM_OPENAI_API_VERSION", "2024-08-01-preview"),
      subscription_key=os.getenv("AZURE_APIM_OPENAI_SUBSCRIPTION_KEY"),
    ),
    kucoin=KucoinConfig(
      api_key=os.getenv("KUCOIN_API_KEY", ""),
      secret=os.getenv("KUCOIN_API_SECRET", ""),
      passphrase=os.getenv("KUCOIN_API_PASSPHRASE", ""),
      base_url=kucoin_base_url,
    ),
    kucoin_futures=KucoinFuturesConfig(
      enabled=_as_bool(os.getenv("KUCOIN_FUTURES_ENABLED"), True),
      base_url=kucoin_futures_base_url,
      margin_mode=os.getenv("KUCOIN_FUTURES_MARGIN_MODE", "cross").strip().lower(),
    ),
    trading=TradingConfig(
      coins=coins,
      flexible_coins_enabled=_as_bool(os.getenv("FLEXIBLE_COINS_ENABLED"), True),
      paper_trading=_as_bool(os.getenv("PAPER_TRADING"), True),
      max_position_usd=float(os.getenv("MAX_POSITION_USD", "500")),
      risk_per_trade_pct=float(os.getenv("RISK_PER_TRADE_PCT", "0.01")),
      min_confidence=float(os.getenv("MIN_CONFIDENCE", "0.65")),
      sentiment_filter_enabled=_as_bool(os.getenv("SENTIMENT_FILTER_ENABLED"), False),
      sentiment_min_score=float(os.getenv("SENTIMENT_MIN_SCORE", "0.55")),
      poll_interval_sec=float(os.getenv("POLL_INTERVAL_SEC", "60")),
      price_change_trigger_pct=float(os.getenv("PRICE_CHANGE_TRIGGER_PCT", "0.5")),
      max_idle_polls=int(os.getenv("MAX_IDLE_POLLS", "10")),
      max_leverage=float(os.getenv("MAX_LEVERAGE", "3")),
      max_trades_per_symbol_per_day=int(os.getenv("MAX_TRADES_PER_SYMBOL_PER_DAY", "6")),
      min_net_profit_usd=float(os.getenv("MIN_NET_PROFIT_USD", "0.5")),
      min_profit_roi_pct=float(os.getenv("MIN_PROFIT_ROI_PCT", "0.008")),
      estimated_slippage_pct=float(os.getenv("ESTIMATED_SLIPPAGE_PCT", "0.001")),
      range_trading_enabled=_as_bool(os.getenv("RANGE_TRADING_ENABLED"), True),
      # Off by default: a 60% tranche at 60% of target compresses a 1.5R admitted setup to
      # only 1.14R gross while the full stop remains. It stays available as an explicit opt-in.
      partial_tp_enabled=_as_bool(os.getenv("PARTIAL_TP_ENABLED"), False),
      kelly_sizing_enabled=_as_bool(os.getenv("KELLY_SIZING_ENABLED"), True),
      kelly_min_trades=int(os.getenv("KELLY_MIN_TRADES", "30")),
      prefer_limit_orders=_as_bool(os.getenv("PREFER_LIMIT_ORDERS"), True),
      limit_order_timeout_sec=float(os.getenv("LIMIT_ORDER_TIMEOUT_SEC", "20")),
      post_loss_cooldown_minutes=float(os.getenv("POST_LOSS_COOLDOWN_MINUTES", "30")),
      max_entry_leverage=float(os.getenv("MAX_ENTRY_LEVERAGE", "3")),
      min_trade_interval_minutes=float(os.getenv("MIN_TRADE_INTERVAL_MINUTES", "10")),
      max_24h_volatility_pct=float(os.getenv("MAX_24H_VOLATILITY_PCT", "25")),
      max_atr_pct_for_entry=float(os.getenv("MAX_ATR_PCT_FOR_ENTRY", "6")),
      entry_limit_expiry_minutes=float(os.getenv("ENTRY_LIMIT_EXPIRY_MINUTES", "30")),
      min_entry_deviation_pct=float(os.getenv("MIN_ENTRY_DEVIATION_PCT", "0.002")),
      research_handoff_after_no_trade_runs=int(os.getenv("RESEARCH_HANDOFF_AFTER_NO_TRADE_RUNS", "6")),
      max_position_equity_pct=float(os.getenv("MAX_POSITION_EQUITY_PCT", "0.5")),
      min_futures_listing_age_days=float(os.getenv("MIN_FUTURES_LISTING_AGE_DAYS", "7")),
      research_handoff_cooldown_min=float(os.getenv("RESEARCH_HANDOFF_COOLDOWN_MIN", "120")),
      min_futures_rr=float(os.getenv("MIN_FUTURES_RR", "1.5")),
      screener_min_turnover_usd_24h=float(os.getenv("SCREENER_MIN_TURNOVER_USD_24H", "5000000")),
      atomic_bracket_enabled=_as_bool(os.getenv("ATOMIC_BRACKET_ENABLED"), True),
      emergency_sl_pct=float(os.getenv("EMERGENCY_SL_PCT", "0.02")),
      flat_agent_cooldown_sec=float(os.getenv("FLAT_AGENT_COOLDOWN_SEC", "3600")),
      active_agent_cooldown_sec=float(os.getenv("ACTIVE_AGENT_COOLDOWN_SEC", "300")),
      flat_backoff_max_multiplier=float(os.getenv("FLAT_BACKOFF_MAX_MULTIPLIER", "4.0")),
      price_noise_multiplier=float(os.getenv("PRICE_NOISE_MULTIPLIER", "4.0")),
      price_trigger_max_multiplier=float(os.getenv("PRICE_TRIGGER_MAX_MULTIPLIER", "2.0")),
    ),
    circuit_breaker=CircuitBreakerConfig(
      max_daily_drawdown_pct=float(os.getenv("CB_MAX_DAILY_DRAWDOWN_PCT", "3.0")),
      max_consecutive_losses=int(os.getenv("CB_MAX_CONSECUTIVE_LOSSES", "3")),
      max_portfolio_heat_pct=float(os.getenv("CB_MAX_PORTFOLIO_HEAT_PCT", "6.0")),
      cooldown_minutes=float(os.getenv("CB_COOLDOWN_MINUTES", "120")),
    ),
    profit_protection=ProfitProtectionConfig(
      enabled=_as_bool(os.getenv("PROFIT_LOCK_ENABLED"), True),
      dry_run=_as_bool(os.getenv("PROFIT_LOCK_DRY_RUN"), False),
      breakeven_trigger_r=float(os.getenv("PROFIT_LOCK_BREAKEVEN_TRIGGER_R", "1.0")),
      breakeven_fee_pct=float(os.getenv("PROFIT_LOCK_BREAKEVEN_FEE_PCT", "0.0015")),
      giveback_pct=float(os.getenv("PROFIT_LOCK_GIVEBACK_PCT", "0.35")),
      min_favorable_excursion_pct=float(os.getenv("PROFIT_LOCK_MIN_FE_PCT", "0.005")),
      no_chase_enabled=_as_bool(os.getenv("NO_CHASE_ENABLED"), True),
      post_win_cooldown_minutes=float(os.getenv("POST_WIN_COOLDOWN_MINUTES", "45")),
      no_chase_buffer_pct=float(os.getenv("NO_CHASE_BUFFER_PCT", "0.001")),
      giveback_arm_r=float(os.getenv("PROFIT_LOCK_GIVEBACK_ARM_R", "1.0")),
      early_cut_enabled=_as_bool(os.getenv("EARLY_CUT_ENABLED"), True),
      early_cut_grace_min=float(os.getenv("EARLY_CUT_GRACE_MIN", "20")),
      early_cut_min_favorable_pct=float(os.getenv("EARLY_CUT_MIN_FAVORABLE_PCT", "0.003")),
      early_cut_mae_frac=float(os.getenv("EARLY_CUT_MAE_FRAC", "0.6")),
    ),
    edge=EdgeConfig(
      enabled=_as_bool(os.getenv("ADAPTIVE_EDGE_ENABLED"), True),
      lookback_trades=int(os.getenv("EDGE_LOOKBACK_TRADES", "30")),
      min_trades=int(os.getenv("EDGE_MIN_TRADES", "8")),
      rr_step=float(os.getenv("EDGE_RR_STEP", "0.5")),
      rr_cap=float(os.getenv("EDGE_RR_CAP", "2.5")),
      rr_stale_hours=float(os.getenv("EDGE_RR_STALE_HOURS", "18")),
      symbol_rr_min_trades=int(os.getenv("EDGE_SYMBOL_RR_MIN_TRADES", "2")),
      bench_lookback=int(os.getenv("EDGE_BENCH_LOOKBACK", "5")),
      bench_min_losses=int(os.getenv("EDGE_BENCH_MIN_LOSSES", "3")),
      bench_cooldown_hours=float(os.getenv("EDGE_BENCH_COOLDOWN_HOURS", "12")),
      bench_cooldown_max_mult=int(os.getenv("EDGE_BENCH_COOLDOWN_MAX_MULT", "4")),
      streak_threshold=int(os.getenv("EDGE_STREAK_THRESHOLD", "2")),
      streak_size_factor=float(os.getenv("EDGE_STREAK_SIZE_FACTOR", "0.5")),
      direction_min_trades=int(os.getenv("EDGE_DIRECTION_MIN_TRADES", "5")),
      negative_expectancy_size_factor=float(os.getenv("EDGE_NEGATIVE_EXPECTANCY_SIZE_FACTOR", "0.5")),
    ),
    regime=RegimeConfig(
      throttle_enabled=_as_bool(os.getenv("REGIME_THROTTLE_ENABLED"), True),
      caution_min_confidence=float(os.getenv("REGIME_CAUTION_MIN_CONFIDENCE", "0.75")),
      caution_size_factor=float(os.getenv("REGIME_CAUTION_SIZE_FACTOR", "0.6")),
      trend_shorts_enabled=_as_bool(os.getenv("TREND_ALIGNED_SHORTS_ENABLED"), True),
      trend_short_min_confidence=float(os.getenv("TREND_SHORT_MIN_CONFIDENCE", "0.78")),
      trend_short_require_15m=_as_bool(os.getenv("TREND_SHORT_REQUIRE_15M"), True),
      alt_long_block_enabled=_as_bool(os.getenv("ALT_LONG_BLOCK_WHEN_BTC_BEARISH"), True),
      alt_majors=tuple(s.strip().upper() for s in os.getenv("ALT_MAJORS", "BTC,ETH").split(",") if s.strip()),
      relative_strength_longs_enabled=_as_bool(os.getenv("RELATIVE_STRENGTH_LONGS_ENABLED"), True),
      relative_strength_min_confidence=float(os.getenv("RELATIVE_STRENGTH_MIN_CONFIDENCE", "0.82")),
      relative_strength_size_factor=float(os.getenv("RELATIVE_STRENGTH_SIZE_FACTOR", "0.5")),
      conviction_sizing_enabled=_as_bool(os.getenv("CONVICTION_SIZING_ENABLED"), True),
      conviction_full_confidence=float(os.getenv("CONVICTION_FULL_CONFIDENCE", "0.85")),
      conviction_min_size_factor=float(os.getenv("CONVICTION_MIN_SIZE_FACTOR", "0.5")),
      deadlock_break_enabled=_as_bool(os.getenv("DEADLOCK_BREAK_ENABLED"), True),
      deadlock_min_confidence=float(os.getenv("DEADLOCK_MIN_CONFIDENCE", "0.72")),
      reversal_longs_enabled=_as_bool(os.getenv("REVERSAL_LONGS_ENABLED"), True),
      reversal_long_min_confidence=float(os.getenv("REVERSAL_LONG_MIN_CONFIDENCE", "0.80")),
      reversal_long_require_15m=_as_bool(os.getenv("REVERSAL_LONG_REQUIRE_15M"), True),
      reversal_shorts_enabled=_as_bool(os.getenv("REVERSAL_SHORTS_ENABLED"), True),
      reversal_short_min_confidence=float(os.getenv("REVERSAL_SHORT_MIN_CONFIDENCE", "0.80")),
      reversal_short_require_15m=_as_bool(os.getenv("REVERSAL_SHORT_REQUIRE_15M"), True),
    ),
    langsmith=LangsmithConfig(
      enabled=_as_bool(os.getenv("LANGSMITH_ENABLED"), False),
      api_key=os.getenv("LANGSMITH_API_KEY"),
      project=os.getenv("LANGSMITH_PROJECT"),
      api_url=os.getenv("LANGSMITH_API_URL"),
      tracing=_as_bool(os.getenv("LANGSMITH_TRACING"), False),
      sample_rate=float(os.getenv("LANGSMITH_SAMPLE_RATE", "0.1")),
    ),
    telegram=TelegramConfig(
      enabled=_as_bool(os.getenv("TELEGRAM_ENABLED"), False),
      bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
      chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
      silent=_as_bool(os.getenv("TELEGRAM_SILENT"), False),
    ),
    supervisor=SupervisorConfig(
      enabled=_as_bool(os.getenv("SUPERVISOR_ENABLED"), True),
      log_file=os.getenv("LOG_FILE", "traide.log"),
      log_max_bytes=int(os.getenv("LOG_MAX_BYTES", "5242880")),
      log_backup_count=int(os.getenv("LOG_BACKUP_COUNT", "3")),
    ),
    dashboard=DashboardConfig(
      enabled=_as_bool(os.getenv("DASHBOARD_PUBLISH_ENABLED"), False),
      connection_string=os.getenv("connection_string", ""),
      table_name=os.getenv("TRAIDE_TABLE_NAME", "traidedashboard"),
      container_name=os.getenv("TRAIDE_BLOB_NAME", "traide-dashboard"),
      publish_interval_sec=float(os.getenv("DASHBOARD_PUBLISH_INTERVAL_SEC", "300")),
      disclosure=os.getenv("DASHBOARD_DISCLOSURE", "normalized").strip().lower(),
      feed_limit=int(os.getenv("DASHBOARD_FEED_LIMIT", "30")),
      index_base=float(os.getenv("DASHBOARD_INDEX_BASE", "100")),
    ),
    tracing_enabled=_as_bool(os.getenv("ENABLE_TRACING"), False),
    console_tracing=_as_bool(os.getenv("ENABLE_CONSOLE_TRACING"), False),
    openai_trace_api_key=os.getenv("OPENAI_TRACE_API_KEY"),
    memory_file=os.getenv("MEMORY_FILE", ".agent_memory.json"),
    retention_days=int(os.getenv("RETENTION_DAYS", "90")),
    agent_max_turns=int(os.getenv("AGENT_MAX_TURNS", "20")),
  )

  validate_config(config)
  return config


def validate_config(cfg: AppConfig) -> None:
  missing: list[str] = []
  if not cfg.azure.endpoint:
    missing.append("AZURE_OPENAI_ENDPOINT")
  if not cfg.azure.deployment:
    missing.append("AZURE_OPENAI_DEPLOYMENT")
  if not cfg.azure.api_key:
    missing.append("AZURE_OPENAI_API_KEY")
  if cfg.apim.subscription_key:
    if not cfg.apim.endpoint:
      missing.append("AZURE_APIM_OPENAI_ENDPOINT")
    if not cfg.apim.deployment:
      missing.append("AZURE_APIM_OPENAI_DEPLOYMENT")
  if not cfg.kucoin.api_key:
    missing.append("KUCOIN_API_KEY")
  if not cfg.kucoin.secret:
    missing.append("KUCOIN_API_SECRET")
  if not cfg.kucoin.passphrase:
    missing.append("KUCOIN_API_PASSPHRASE")
  if not cfg.trading.coins:
    missing.append("COINS (e.g., BTC-USDT)")

  if missing:
    raise ValueError(
      f"Missing required configuration: {', '.join(missing)}. Fill .env or environment variables."
    )

  invalid: list[str] = []
  if not (0.0 <= cfg.trading.min_confidence <= 1.0):
    invalid.append(f"MIN_CONFIDENCE={cfg.trading.min_confidence} (must be 0.0–1.0)")
  if not (0 < cfg.trading.max_leverage <= 125):
    invalid.append(f"MAX_LEVERAGE={cfg.trading.max_leverage} (must be >0 and <=125)")
  if cfg.trading.poll_interval_sec <= 0:
    invalid.append(f"POLL_INTERVAL_SEC={cfg.trading.poll_interval_sec} (must be >0)")
  if cfg.trading.max_position_usd <= 0:
    invalid.append(f"MAX_POSITION_USD={cfg.trading.max_position_usd} (must be >0)")
  if not (0 < cfg.trading.risk_per_trade_pct <= 1.0):
    invalid.append(f"RISK_PER_TRADE_PCT={cfg.trading.risk_per_trade_pct} (must be >0 and <=1.0)")
  if cfg.trading.flat_agent_cooldown_sec < 0 or cfg.trading.active_agent_cooldown_sec < 0:
    invalid.append("FLAT_AGENT_COOLDOWN_SEC and ACTIVE_AGENT_COOLDOWN_SEC must be >=0")
  if not (cfg.trading.flat_backoff_max_multiplier >= 1.0):
    invalid.append(
      f"FLAT_BACKOFF_MAX_MULTIPLIER={cfg.trading.flat_backoff_max_multiplier} (must be >=1.0)"
    )
  if cfg.trading.price_noise_multiplier < 0:
    invalid.append(f"PRICE_NOISE_MULTIPLIER={cfg.trading.price_noise_multiplier} (must be >=0)")
  if cfg.trading.price_trigger_max_multiplier < 1.0:
    invalid.append(f"PRICE_TRIGGER_MAX_MULTIPLIER={cfg.trading.price_trigger_max_multiplier} (must be >=1.0 so the ceiling never drops below the base trigger)")
  if cfg.edge.direction_min_trades < 1:
    invalid.append(f"EDGE_DIRECTION_MIN_TRADES={cfg.edge.direction_min_trades} (must be >=1)")
  if not (0.0 < cfg.edge.negative_expectancy_size_factor <= 1.0):
    invalid.append(
      f"EDGE_NEGATIVE_EXPECTANCY_SIZE_FACTOR={cfg.edge.negative_expectancy_size_factor} (must be >0 and <=1.0)"
    )
  if not (0.0 <= cfg.regime.relative_strength_min_confidence <= 1.0):
    invalid.append(f"RELATIVE_STRENGTH_MIN_CONFIDENCE={cfg.regime.relative_strength_min_confidence} (must be 0.0–1.0)")
  if not (0.0 < cfg.regime.relative_strength_size_factor <= 1.0):
    invalid.append(f"RELATIVE_STRENGTH_SIZE_FACTOR={cfg.regime.relative_strength_size_factor} (must be >0 and <=1.0)")
  if not (0.0 <= cfg.circuit_breaker.max_portfolio_heat_pct <= 100.0):
    invalid.append(f"CB_MAX_PORTFOLIO_HEAT_PCT={cfg.circuit_breaker.max_portfolio_heat_pct} (must be 0–100)")

  if invalid:
    raise ValueError(
      f"Invalid configuration values: {', '.join(invalid)}."
    )
