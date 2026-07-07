"""Tests for the KuCoin st-orders bracket trigger mapping (src/kucoin.py bracket_trigger_prices)."""

from src.kucoin import bracket_trigger_prices


def test_long_bracket_mapping():
    # Long: TP above entry -> up trigger; SL below entry -> down trigger.
    out = bracket_trigger_prices("buy", take_profit_price=110.0, stop_loss_price=95.0)
    assert out == {"triggerStopUpPrice": 110.0, "triggerStopDownPrice": 95.0}


def test_short_bracket_mapping():
    # Short: TP below entry -> down trigger; SL above entry -> up trigger (the direction flip).
    out = bracket_trigger_prices("sell", take_profit_price=90.0, stop_loss_price=105.0)
    assert out == {"triggerStopUpPrice": 105.0, "triggerStopDownPrice": 90.0}


def test_mapping_accepts_strings_and_casts():
    out = bracket_trigger_prices("buy", "110", "95")
    assert out["triggerStopUpPrice"] == 110.0 and out["triggerStopDownPrice"] == 95.0
