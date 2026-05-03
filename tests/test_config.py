import pytest
from src.config import AppConfig, AzureConfig, ApimConfig, KucoinConfig, KucoinFuturesConfig, TradingConfig, LangsmithConfig, TelegramConfig, SupervisorConfig, validate_config


def _make_valid_config(**overrides) -> AppConfig:
    trading_kwargs = dict(
        coins=["BTC-USDT"],
        flexible_coins_enabled=True,
        paper_trading=True,
        max_position_usd=100.0,
        risk_per_trade_pct=0.01,
        min_confidence=0.6,
        sentiment_filter_enabled=False,
        sentiment_min_score=0.55,
        poll_interval_sec=30.0,
        price_change_trigger_pct=0.5,
        max_idle_polls=10,
        max_leverage=3.0,
        max_trades_per_symbol_per_day=10,
        min_net_profit_usd=1.5,
        min_profit_roi_pct=0.008,
        estimated_slippage_pct=0.0005,
    )
    trading_kwargs.update(overrides)
    return AppConfig(
        azure=AzureConfig(endpoint="https://x.openai.azure.com/", deployment="gpt-5.2", api_version="2024-10-01-preview", api_key="key"),
        apim=ApimConfig(endpoint="", deployment="", api_version="", subscription_key=None),
        kucoin=KucoinConfig(api_key="k", secret="s", passphrase="p", base_url="https://api.kucoin.com"),
        kucoin_futures=KucoinFuturesConfig(enabled=False, base_url="https://api-futures.kucoin.com"),
        trading=TradingConfig(**trading_kwargs),
        langsmith=LangsmithConfig(enabled=False, api_key=None, project=None, api_url=None, tracing=False),
        telegram=TelegramConfig(enabled=False, bot_token="", chat_id="", silent=False),
        supervisor=SupervisorConfig(enabled=False, log_file="traide.log", log_max_bytes=5242880, log_backup_count=3),
        tracing_enabled=False,
        console_tracing=False,
        openai_trace_api_key=None,
        memory_file=".agent_memory.json",
        retention_days=7,
        agent_max_turns=100,
    )


def test_valid_config_passes():
    cfg = _make_valid_config()
    validate_config(cfg)  # should not raise


def test_missing_azure_key():
    cfg = _make_valid_config()
    cfg.azure.api_key = ""
    with pytest.raises(ValueError, match="AZURE_OPENAI_API_KEY"):
        validate_config(cfg)


def test_missing_kucoin_key():
    cfg = _make_valid_config()
    cfg.kucoin.api_key = ""
    with pytest.raises(ValueError, match="KUCOIN_API_KEY"):
        validate_config(cfg)


def test_missing_coins():
    cfg = _make_valid_config()
    cfg.trading.coins = []
    with pytest.raises(ValueError, match="COINS"):
        validate_config(cfg)


def test_invalid_min_confidence_too_high():
    cfg = _make_valid_config(min_confidence=1.5)
    with pytest.raises(ValueError, match="MIN_CONFIDENCE"):
        validate_config(cfg)


def test_invalid_min_confidence_negative():
    cfg = _make_valid_config(min_confidence=-0.1)
    with pytest.raises(ValueError, match="MIN_CONFIDENCE"):
        validate_config(cfg)


def test_invalid_max_leverage_zero():
    cfg = _make_valid_config(max_leverage=0.0)
    with pytest.raises(ValueError, match="MAX_LEVERAGE"):
        validate_config(cfg)


def test_invalid_max_leverage_too_high():
    cfg = _make_valid_config(max_leverage=200.0)
    with pytest.raises(ValueError, match="MAX_LEVERAGE"):
        validate_config(cfg)


def test_invalid_poll_interval():
    cfg = _make_valid_config(poll_interval_sec=0.0)
    with pytest.raises(ValueError, match="POLL_INTERVAL_SEC"):
        validate_config(cfg)


def test_invalid_max_position_usd():
    cfg = _make_valid_config(max_position_usd=0.0)
    with pytest.raises(ValueError, match="MAX_POSITION_USD"):
        validate_config(cfg)
