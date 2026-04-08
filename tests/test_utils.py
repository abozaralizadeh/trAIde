import pytest
from src.utils import normalize_symbol


def test_already_normalized():
    assert normalize_symbol("BTC-USDT") == "BTC-USDT"
    assert normalize_symbol("ETH-USDT") == "ETH-USDT"


def test_plain_concat():
    assert normalize_symbol("BTCUSDT") == "BTC-USDT"
    assert normalize_symbol("ETHUSDT") == "ETH-USDT"
    assert normalize_symbol("SOLUSDT") == "SOL-USDT"


def test_futures_symbol():
    assert normalize_symbol("XBTUSDTM") == "BTC-USDT"
    assert normalize_symbol("ETHUSDTM") == "ETH-USDT"


def test_lowercase_input():
    assert normalize_symbol("btcusdt") == "BTC-USDT"
    assert normalize_symbol("btc-usdt") == "BTC-USDT"


def test_empty_input():
    assert normalize_symbol("") == ""
    assert normalize_symbol(None) == ""


def test_whitespace_stripped():
    assert normalize_symbol("  BTC-USDT  ") == "BTC-USDT"
