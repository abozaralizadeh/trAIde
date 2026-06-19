import queue
from unittest.mock import patch, MagicMock

import pytest

from src.config import (
  AppConfig, AzureConfig, ApimConfig, KucoinConfig, KucoinFuturesConfig,
  TradingConfig, CircuitBreakerConfig, ProfitProtectionConfig, RegimeConfig, LangsmithConfig,
  TelegramConfig, SupervisorConfig, DashboardConfig,
)
from src.telegram import (
  TelegramNotifier, _esc, _fmt_price, _format_orders, _split_message, _strip_tags,
  markdown_to_telegram_html, MAX_MESSAGE_LENGTH,
)


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
    range_trading_enabled=True,
    partial_tp_enabled=True,
    kelly_sizing_enabled=True,
    kelly_min_trades=30,
    prefer_limit_orders=True,
    limit_order_timeout_sec=20.0,
    post_loss_cooldown_minutes=30.0,
    max_entry_leverage=3.0,
    min_trade_interval_minutes=5.0,
    max_24h_volatility_pct=25.0,
    max_atr_pct_for_entry=5.0,
    entry_limit_expiry_minutes=30.0,
    min_entry_deviation_pct=0.002,
    research_handoff_after_no_trade_runs=3,
  )
  return AppConfig(
    azure=AzureConfig(endpoint="https://x.openai.azure.com/", deployment="gpt-5.2", api_version="2024-10-01-preview", api_key="key"),
    apim=ApimConfig(endpoint="", deployment="", api_version="", subscription_key=None),
    kucoin=KucoinConfig(api_key="k", secret="s", passphrase="p", base_url="https://api.kucoin.com"),
    kucoin_futures=KucoinFuturesConfig(enabled=True, base_url="https://api-futures.kucoin.com"),
    trading=TradingConfig(**trading_kwargs),
    circuit_breaker=CircuitBreakerConfig(max_daily_drawdown_pct=8.0, max_consecutive_losses=4, max_portfolio_heat_pct=20.0, cooldown_minutes=120.0),
    profit_protection=ProfitProtectionConfig(enabled=True, dry_run=False, breakeven_trigger_r=1.0, breakeven_fee_pct=0.0015, giveback_pct=0.35, min_favorable_excursion_pct=0.005, no_chase_enabled=True, post_win_cooldown_minutes=45.0, no_chase_buffer_pct=0.001),
    regime=RegimeConfig(throttle_enabled=True, caution_min_confidence=0.75, caution_size_factor=0.6, trend_shorts_enabled=True, trend_short_min_confidence=0.78, trend_short_require_15m=True),
    langsmith=LangsmithConfig(enabled=False, api_key=None, project=None, api_url=None, tracing=False),
    telegram=TelegramConfig(enabled=telegram_enabled, bot_token=bot_token, chat_id=chat_id, silent=silent),
    supervisor=SupervisorConfig(enabled=False, log_file="traide.log", log_max_bytes=5242880, log_backup_count=3),
    dashboard=DashboardConfig(enabled=False, connection_string="", table_name="traidedashboard", container_name="traide-dashboard", publish_interval_sec=300.0, disclosure="normalized", feed_limit=30, index_base=100.0),
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


class TestMarkdownToTelegramHtml:
  def test_bold_and_italic(self):
    assert markdown_to_telegram_html("**big** and *small*") == "<b>big</b> and <i>small</i>"

  def test_underscore_bold_and_strike(self):
    assert markdown_to_telegram_html("__b__ ~~gone~~") == "<b>b</b> <s>gone</s>"

  def test_heading_becomes_bold_line(self):
    assert markdown_to_telegram_html("## STEP 3 — Trade") == "<b>STEP 3 — Trade</b>"

  def test_bullets_become_dots(self):
    out = markdown_to_telegram_html("- one\n- two")
    assert out == "• one\n• two"

  def test_inline_code(self):
    assert markdown_to_telegram_html("set TP at `68,800`") == "set TP at <code>68,800</code>"

  def test_html_special_chars_escaped(self):
    # RSI <20 & >80 must be escaped so Telegram does not choke on the entities.
    out = markdown_to_telegram_html("RSI <20 & >80")
    assert out == "RSI &lt;20 &amp; &gt;80"

  def test_snake_case_not_italicized(self):
    # Tool names with underscores must survive untouched (a common false-italic trap).
    assert markdown_to_telegram_html("call place_market_order now") == "call place_market_order now"

  def test_arithmetic_asterisks_untouched(self):
    assert markdown_to_telegram_html("5*3=15 and a*b*c") == "5*3=15 and a*b*c"

  def test_link_conversion(self):
    out = markdown_to_telegram_html("see [CoinDesk](https://x.io/a)")
    assert out == 'see <a href="https://x.io/a">CoinDesk</a>'

  def test_code_fence_no_newline_spanning_tag(self):
    # Each code line is wrapped on its own so message-splitting can never break a tag.
    out = markdown_to_telegram_html("```\nfoo()\nbar()\n```")
    assert out == "<code>foo()</code>\n<code>bar()</code>"

  def test_empty(self):
    assert markdown_to_telegram_html("") == ""

  def test_strip_tags_roundtrip(self):
    html = markdown_to_telegram_html("## Title\n**bold** with `code` and RSI <20")
    plain = _strip_tags(html)
    assert "<" not in plain.replace("<20", "X")  # only the literal <20 remains, no tags
    assert "Title" in plain and "bold" in plain and "code" in plain


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

  def test_notify_agent_run_renders_handoffs_and_research(self):
    cfg = _make_cfg(telegram_enabled=False)
    notifier = TelegramNotifier(cfg)
    notifier.enabled = True
    messages = []
    notifier.send = lambda text: messages.append(text)

    result = {
      "narrative": "## Summary\n**Refreshed** the coin list and entered ETH.",
      "decisions": ["live order: long ETH-USDT (orderId=z9)"],
      "tool_results": [],
      "handoffs": [
        {"from": "Trading Agent", "to": "Research Agent"},
        {"from": "Research Agent", "to": "Trading Agent"},
      ],
      "research": ["added coin AVAX-USDT — fresh breakout", "removed coin DOGE-USDT — no catalyst"],
      "agentsUsed": ["Research Agent", "Trading Agent"],
    }
    notifier.notify_agent_run(["idle_threshold"], result)
    assert len(messages) == 1
    msg = messages[0]
    assert "Handoffs" in msg
    assert "Trading Agent → Research Agent" in msg
    assert "Research Agent" in msg
    assert "AVAX-USDT" in msg
    assert "removed coin DOGE-USDT" in msg
    # The markdown narrative is converted to HTML, not shown with literal ## / **.
    assert "<b>Summary</b>" in msg
    assert "<b>Refreshed</b>" in msg
    assert "## Summary" not in msg

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


class TestSplitMessage:
  def test_short_message_not_split(self):
    chunks = _split_message("hello")
    assert chunks == ["hello"]

  def test_exact_limit_not_split(self):
    text = "x" * MAX_MESSAGE_LENGTH
    chunks = _split_message(text)
    assert len(chunks) == 1
    assert chunks[0] == text

  def test_long_message_split_at_newlines(self):
    line = "a" * 100 + "\n"
    text = line * 50  # 5050 chars, > 4096
    chunks = _split_message(text)
    assert len(chunks) >= 2
    for chunk in chunks:
      assert len(chunk) <= MAX_MESSAGE_LENGTH
    reassembled = "\n".join(chunks)
    assert reassembled.replace("\n", "") == text.replace("\n", "")

  def test_long_message_hard_split_when_no_newlines(self):
    text = "x" * (MAX_MESSAGE_LENGTH * 2 + 100)
    chunks = _split_message(text)
    assert len(chunks) == 3
    for chunk in chunks:
      assert len(chunk) <= MAX_MESSAGE_LENGTH
    assert "".join(chunks) == text

  def test_send_raw_sends_all_chunks(self):
    cfg = _make_cfg(telegram_enabled=False)
    notifier = TelegramNotifier(cfg)
    long_text = ("x" * 100 + "\n") * 50  # > 4096 chars
    with patch("src.telegram.requests.post") as mock_post:
      mock_post.return_value = MagicMock(status_code=200)
      notifier.token = "tok"
      notifier._send_raw(long_text)
      assert mock_post.call_count >= 2
      for call in mock_post.call_args_list:
        sent_text = call.kwargs["json"]["text"] if "json" in call.kwargs else call[1]["json"]["text"]
        assert len(sent_text) <= MAX_MESSAGE_LENGTH


class TestNarrativeNotTruncated:
  def test_full_narrative_sent(self):
    cfg = _make_cfg(telegram_enabled=False)
    notifier = TelegramNotifier(cfg)
    notifier.enabled = True
    messages = []
    notifier.send = lambda text: messages.append(text)
    long_narrative = "word " * 200  # 1000 chars
    result = {"narrative": long_narrative, "decisions": [], "tool_results": []}
    notifier.notify_agent_run(["idle"], result)
    full_text = "".join(messages)
    assert "...truncated" not in full_text
    assert "word " * 199 in full_text


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
