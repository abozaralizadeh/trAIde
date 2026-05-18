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


@dataclass
class CircuitBreakerConfig:
  max_daily_drawdown_pct: float
  max_consecutive_losses: int
  max_portfolio_heat_pct: float
  cooldown_minutes: float


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


@dataclass
class AppConfig:
  azure: AzureConfig
  apim: ApimConfig
  kucoin: KucoinConfig
  kucoin_futures: KucoinFuturesConfig
  trading: TradingConfig
  circuit_breaker: CircuitBreakerConfig
  langsmith: LangsmithConfig
  telegram: TelegramConfig
  supervisor: SupervisorConfig
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
    ),
    trading=TradingConfig(
      coins=coins,
      flexible_coins_enabled=_as_bool(os.getenv("FLEXIBLE_COINS_ENABLED"), True),
      paper_trading=_as_bool(os.getenv("PAPER_TRADING"), True),
      max_position_usd=float(os.getenv("MAX_POSITION_USD", "500")),
      risk_per_trade_pct=float(os.getenv("RISK_PER_TRADE_PCT", "0.10")),
      min_confidence=float(os.getenv("MIN_CONFIDENCE", "0.65")),
      sentiment_filter_enabled=_as_bool(os.getenv("SENTIMENT_FILTER_ENABLED"), False),
      sentiment_min_score=float(os.getenv("SENTIMENT_MIN_SCORE", "0.55")),
      poll_interval_sec=float(os.getenv("POLL_INTERVAL_SEC", "60")),
      price_change_trigger_pct=float(os.getenv("PRICE_CHANGE_TRIGGER_PCT", "0.5")),
      max_idle_polls=int(os.getenv("MAX_IDLE_POLLS", "10")),
      max_leverage=float(os.getenv("MAX_LEVERAGE", "3")),
      max_trades_per_symbol_per_day=int(os.getenv("MAX_TRADES_PER_SYMBOL_PER_DAY", "10")),
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
      min_trade_interval_minutes=float(os.getenv("MIN_TRADE_INTERVAL_MINUTES", "5")),
      max_24h_volatility_pct=float(os.getenv("MAX_24H_VOLATILITY_PCT", "25")),
      max_atr_pct_for_entry=float(os.getenv("MAX_ATR_PCT_FOR_ENTRY", "5")),
      entry_limit_expiry_minutes=float(os.getenv("ENTRY_LIMIT_EXPIRY_MINUTES", "30")),
      min_entry_deviation_pct=float(os.getenv("MIN_ENTRY_DEVIATION_PCT", "0.002")),
    ),
    circuit_breaker=CircuitBreakerConfig(
      max_daily_drawdown_pct=float(os.getenv("CB_MAX_DAILY_DRAWDOWN_PCT", "10.0")),
      max_consecutive_losses=int(os.getenv("CB_MAX_CONSECUTIVE_LOSSES", "3")),
      max_portfolio_heat_pct=float(os.getenv("CB_MAX_PORTFOLIO_HEAT_PCT", "20.0")),
      cooldown_minutes=float(os.getenv("CB_COOLDOWN_MINUTES", "120")),
    ),
    langsmith=LangsmithConfig(
      enabled=_as_bool(os.getenv("LANGSMITH_ENABLED"), False),
      api_key=os.getenv("LANGSMITH_API_KEY"),
      project=os.getenv("LANGSMITH_PROJECT"),
      api_url=os.getenv("LANGSMITH_API_URL"),
      tracing=_as_bool(os.getenv("LANGSMITH_TRACING"), False),
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
