from __future__ import annotations


def normalize_symbol(sym: str) -> str:
    """Normalize a trading symbol to KuCoin dash format (e.g., BTCUSDT -> BTC-USDT, XBTUSDTM -> BTC-USDT)."""
    s = (sym or "").strip().upper()
    if not s:
        return s
    if "-" in s:
        return s
    if s.endswith("M"):
        base_quote = s[:-1]
        if base_quote.startswith("XBT"):
            base_quote = "BTC" + base_quote[3:]
        if base_quote.endswith("USDT") and len(base_quote) > 4:
            return f"{base_quote[:-4]}-USDT"
    if s.endswith("USDT") and len(s) > 4:
        return f"{s[:-4]}-USDT"
    return s
