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
  max_daily_drawdown_pct: float
  reset_drawdown_on_start: bool


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
  langsmith: LangsmithConfig
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
      max_position_usd=float(os.getenv("MAX_POSITION_USD", "100")),
      risk_per_trade_pct=float(os.getenv("RISK_PER_TRADE_PCT", "0.01")),
      min_confidence=float(os.getenv("MIN_CONFIDENCE", "0.6")),
      sentiment_filter_enabled=_as_bool(os.getenv("SENTIMENT_FILTER_ENABLED"), False),
      sentiment_min_score=float(os.getenv("SENTIMENT_MIN_SCORE", "0.55")),
      poll_interval_sec=float(os.getenv("POLL_INTERVAL_SEC", "30")),
      price_change_trigger_pct=float(os.getenv("PRICE_CHANGE_TRIGGER_PCT", "0.5")),
      max_idle_polls=int(os.getenv("MAX_IDLE_POLLS", "10")),
      max_leverage=float(os.getenv("MAX_LEVERAGE", "3")),
      max_trades_per_symbol_per_day=int(os.getenv("MAX_TRADES_PER_SYMBOL_PER_DAY", "3")),
      max_daily_drawdown_pct=float(os.getenv("MAX_DAILY_DRAWDOWN_PCT", "8")),
      reset_drawdown_on_start=_as_bool(os.getenv("RESET_DRAWDOWN_ON_START"), False),
    ),
    langsmith=LangsmithConfig(
      enabled=_as_bool(os.getenv("LANGSMITH_ENABLED"), False),
      api_key=os.getenv("LANGSMITH_API_KEY"),
      project=os.getenv("LANGSMITH_PROJECT"),
      api_url=os.getenv("LANGSMITH_API_URL"),
      tracing=_as_bool(os.getenv("LANGSMITH_TRACING"), False),
    ),
    tracing_enabled=_as_bool(os.getenv("ENABLE_TRACING"), False),
    console_tracing=_as_bool(os.getenv("ENABLE_CONSOLE_TRACING"), False),
    openai_trace_api_key=os.getenv("OPENAI_TRACE_API_KEY"),
    memory_file=os.getenv("MEMORY_FILE", ".agent_memory.json"),
    retention_days=int(os.getenv("RETENTION_DAYS", "7")),
    agent_max_turns=int(os.getenv("AGENT_MAX_TURNS", "50")),
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
