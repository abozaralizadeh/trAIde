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
  paper_trading: bool
  max_position_usd: float
  min_confidence: float
  poll_interval_sec: float
  price_change_trigger_pct: float
  max_idle_polls: int
  max_leverage: float


@dataclass
class AppConfig:
  azure: AzureConfig
  apim: ApimConfig
  kucoin: KucoinConfig
  kucoin_futures: KucoinFuturesConfig
  trading: TradingConfig
  tracing_enabled: bool
  console_tracing: bool
  openai_trace_api_key: Optional[str]
  memory_file: str


def load_config() -> AppConfig:
  coins_env = os.getenv("COINS", "")
  coins = [c.strip() for c in coins_env.split(",") if c.strip()]
  kucoin_base_url = os.getenv("KUCOIN_BASE_URL", "https://api.kucoin.com")

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
      base_url=kucoin_base_url,
    ),
    trading=TradingConfig(
      coins=coins,
      paper_trading=_as_bool(os.getenv("PAPER_TRADING"), True),
      max_position_usd=float(os.getenv("MAX_POSITION_USD", "100")),
      min_confidence=float(os.getenv("MIN_CONFIDENCE", "0.6")),
      poll_interval_sec=float(os.getenv("POLL_INTERVAL_SEC", "30")),
      price_change_trigger_pct=float(os.getenv("PRICE_CHANGE_TRIGGER_PCT", "0.5")),
      max_idle_polls=int(os.getenv("MAX_IDLE_POLLS", "10")),
      max_leverage=float(os.getenv("MAX_LEVERAGE", "3")),
    ),
    tracing_enabled=_as_bool(os.getenv("ENABLE_TRACING"), False),
    console_tracing=_as_bool(os.getenv("ENABLE_CONSOLE_TRACING"), False),
    openai_trace_api_key=os.getenv("OPENAI_TRACE_API_KEY"),
    memory_file=os.getenv("MEMORY_FILE", ".agent_memory.json"),
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
