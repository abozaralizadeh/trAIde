import queue
from unittest.mock import patch, MagicMock

import pytest

from src.config import (
  AppConfig, AzureConfig, ApimConfig, KucoinConfig, KucoinFuturesConfig,
  TradingConfig, LangsmithConfig, TelegramConfig,
)
from src.telegram import TelegramNotifier, _esc, _fmt_price, _format_orders, MAX_MESSAGE_LENGTH


def _make_cfg(telegram_enabled=False, bot_token="tok123", chat_id="456", silent=False, **overrides) -> AppConfig:
  trading_kwargs = dict(
    coins=["BTC-USDT", "ETH-USDT"],
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
  return AppConfig(
    azure=AzureConfig(endpoint="https://x.openai.azure.com/", deployment="gpt-5.2", api_version="2024-10-01-preview", api_key="key"),
    apim=ApimConfig(endpoint="", deployment="", api_version="", subscription_key=None),
    kucoin=KucoinConfig(api_key="k", secret="s", passphrase="p", base_url="https://api.kucoin.com"),
    kucoin_futures=KucoinFuturesConfig(enabled=True, base_url="https://api-futures.kucoin.com"),
    trading=TradingConfig(**trading_kwargs),
    langsmith=LangsmithConfig(enabled=False, api_key=None, project=None, api_url=None, tracing=False),
    telegram=TelegramConfig(enabled=telegram_enabled, bot_token=bot_token, chat_id=chat_id, silent=silent),
    tracing_enabled=False,
    console_tracing=False,
    openai_trace_api_key=None,
    memory_file=".agent_memory.json",
    retention_days=7,
    agent_max_turns=100,
  )


class TestTelegramNotifierDisabled:
  def test_disabled_does_not_start_worker(self):
    cfg = _make_cfg(telegram_enabled=False)
    notifier = TelegramNotifier(cfg)
    assert not notifier.enabled
    assert notifier._thread is None

  @patch("src.telegram.requests.post")
  def test_disabled_send_does_nothing(self, mock_post):
    cfg = _make_cfg(telegram_enabled=False)
    notifier = TelegramNotifier(cfg)
    notifier.send("hello")
    mock_post.assert_not_called()

  def test_missing_token_disables(self):
    cfg = _make_cfg(telegram_enabled=True, bot_token="", chat_id="456")
    notifier = TelegramNotifier(cfg)
    assert not notifier.enabled

  def test_missing_chat_id_disables(self):
    cfg = _make_cfg(telegram_enabled=True, bot_token="tok", chat_id="")
    notifier = TelegramNotifier(cfg)
    assert not notifier.enabled


class TestEscaping:
  def test_html_escape(self):
    assert _esc("<b>test</b>") == "&lt;b&gt;test&lt;/b&gt;"
    assert _esc("a & b") == "a &amp; b"

  def test_fmt_price_large(self):
    assert _fmt_price(67234.5) == "$67,234.50"

  def test_fmt_price_small(self):
    assert _fmt_price(0.00042) == "$0.000420"

  def test_fmt_price_medium(self):
    assert _fmt_price(5.123) == "$5.1230"

  def test_fmt_price_invalid(self):
    assert _fmt_price(None) == "None"


class TestFormatOrders:
  def test_live_order(self):
    tool_results = [{
      "orderId": "abc123",
      "side": "buy",
      "symbol": "BTC-USDT",
      "orderRequest": {"symbol": "BTC-USDT", "side": "buy", "funds": "85.00"},
      "tradeRecord": {"price": 67234.5},
      "bracket": {
        "stop": {"stopPrice": 66500.0},
        "takeProfit": {"stopPrice": 68800.0},
      },
      "rr": 2.13,
    }]
    msgs = _format_orders(tool_results)
    assert len(msgs) == 1
    msg = msgs[0]
    assert "BTC-USDT" in msg
    assert "BUY" in msg
    assert "LIVE" in msg
    assert "Stop Loss" in msg
    assert "Take Profit" in msg
    assert "2.13" in msg
    assert "abc123" in msg

  def test_paper_order(self):
    tool_results = [{
      "paper": True,
      "orderRequest": {"symbol": "ETH-USDT", "side": "sell", "size": "0.5"},
      "tradeRecord": {"price": 3450.0},
      "bracket": {},
    }]
    msgs = _format_orders(tool_results)
    assert len(msgs) == 1
    assert "PAPER" in msgs[0]
    assert "ETH-USDT" in msgs[0]
    assert "SELL" in msgs[0]

  def test_transfer(self):
    tool_results = [{
      "transfer": {"orderId": "txn789"},
      "amount": 50.0,
      "currency": "USDT",
      "direction": "futures_to_spot",
    }]
    msgs = _format_orders(tool_results)
    assert len(msgs) == 1
    assert "Transfer" in msgs[0]
    assert "50.0" in msgs[0]

  def test_skipped_not_formatted(self):
    tool_results = [{"skipped": True, "reason": "low confidence"}]
    msgs = _format_orders(tool_results)
    assert len(msgs) == 0

  def test_rejected_not_formatted(self):
    tool_results = [{"rejected": True, "reason": "Exceeds max"}]
    msgs = _format_orders(tool_results)
    assert len(msgs) == 0


class TestNotifierMessages:
  def test_notify_startup_format(self):
    cfg = _make_cfg(telegram_enabled=False)
    notifier = TelegramNotifier(cfg)
    # Call directly to check format (won't send since disabled)
    notifier.enabled = True  # temporarily enable to capture
    messages = []
    notifier.send = lambda text: messages.append(text)
    notifier.notify_startup(cfg)
    assert len(messages) == 1
    msg = messages[0]
    assert "trAIde Bot Started" in msg
    assert "PAPER" in msg
    assert "BTC-USDT" in msg
    assert "$100" in msg

  def test_notify_agent_run_with_decisions(self):
    cfg = _make_cfg(telegram_enabled=False)
    notifier = TelegramNotifier(cfg)
    notifier.enabled = True
    messages = []
    notifier.send = lambda text: messages.append(text)

    result = {
      "narrative": "Analyzed BTC and ETH. BTC shows strong bullish momentum.",
      "decisions": ["live order: buy BTC-USDT (orderId=abc)", "decline: low confidence (conf=0.4)"],
      "tool_results": [{
        "orderId": "abc",
        "side": "buy",
        "symbol": "BTC-USDT",
        "orderRequest": {"symbol": "BTC-USDT", "side": "buy", "funds": "85.00"},
        "tradeRecord": {"price": 67234.5},
        "bracket": {},
      }],
    }
    notifier.notify_agent_run(["price_move:BTC-USDT:0.45%"], result)
    assert len(messages) == 1
    msg = messages[0]
    assert "trAIde Agent Run" in msg
    assert "price_move" in msg
    assert "Decisions" in msg
    assert "Narrative" in msg

  def test_notify_error_format(self):
    cfg = _make_cfg(telegram_enabled=False)
    notifier = TelegramNotifier(cfg)
    notifier.enabled = True
    messages = []
    notifier.send = lambda text: messages.append(text)
    notifier.notify_error("ConnectionError: timeout", context="Agent run")
    assert len(messages) == 1
    msg = messages[0]
    assert "trAIde Error" in msg
    assert "Agent run" in msg
    assert "ConnectionError" in msg


class TestTruncation:
  def test_long_message_truncated(self):
    cfg = _make_cfg(telegram_enabled=False)
    notifier = TelegramNotifier(cfg)
    long_text = "x" * (MAX_MESSAGE_LENGTH + 500)
    result = notifier._send_raw  # we'll test the truncation logic directly
    # Directly check truncation in _send_raw by mocking requests
    with patch("src.telegram.requests.post") as mock_post:
      mock_post.return_value = MagicMock(status_code=200)
      notifier.token = "tok"
      notifier._send_raw(long_text)
      call_args = mock_post.call_args
      sent_text = call_args.kwargs["json"]["text"] if "json" in call_args.kwargs else call_args[1]["json"]["text"]
      assert len(sent_text) <= MAX_MESSAGE_LENGTH
      assert "truncated" in sent_text


class TestQueueFull:
  def test_queue_full_drops_gracefully(self):
    cfg = _make_cfg(telegram_enabled=True, bot_token="tok", chat_id="123")
    with patch.object(TelegramNotifier, "_start_worker"):
      notifier = TelegramNotifier(cfg)
    # Fill the queue
    for i in range(100):
      notifier._queue.put_nowait(f"msg{i}")
    # Next send should drop silently
    notifier.send("overflow")
    assert notifier._queue.qsize() == 100
