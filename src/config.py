import os
from dataclasses import dataclass
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
  research_handoff_cooldown_min: float = 30.0 # min minutes between forced Research handoffs (0=off)


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
      risk_per_trade_pct=float(os.getenv("RISK_PER_TRADE_PCT", "0.02")),
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
      partial_tp_enabled=_as_bool(os.getenv("PARTIAL_TP_ENABLED"), True),
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
      research_handoff_after_no_trade_runs=int(os.getenv("RESEARCH_HANDOFF_AFTER_NO_TRADE_RUNS", "3")),
      max_position_equity_pct=float(os.getenv("MAX_POSITION_EQUITY_PCT", "0.5")),
      min_futures_listing_age_days=float(os.getenv("MIN_FUTURES_LISTING_AGE_DAYS", "7")),
      research_handoff_cooldown_min=float(os.getenv("RESEARCH_HANDOFF_COOLDOWN_MIN", "30")),
    ),
    circuit_breaker=CircuitBreakerConfig(
      max_daily_drawdown_pct=float(os.getenv("CB_MAX_DAILY_DRAWDOWN_PCT", "5.0")),
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
    retention_days=int(os.getenv("RETENTION_DAYS", "14")),
    agent_max_turns=int(os.getenv("AGENT_MAX_TURNS", "100")),
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

  if invalid:
    raise ValueError(
      f"Invalid configuration values: {', '.join(invalid)}."
    )
