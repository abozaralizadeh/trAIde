from src.tools import normalize_futures_side, round_price_to_tick


def test_futures_side_aliases_are_normalized_for_kucoin():
    assert normalize_futures_side("long") == "buy"
    assert normalize_futures_side("BUY") == "buy"
    assert normalize_futures_side("short") == "sell"
    assert normalize_futures_side("sell") == "sell"
    assert normalize_futures_side("hold") is None


def test_price_rounding_uses_contract_tick_without_float_noise():
    assert round_price_to_tick(3.5856, 0.001) == 3.586
    assert round_price_to_tick(100.13, 0.25) == 100.25
