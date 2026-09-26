from __future__ import annotations


# A spot balance worth less than this (USD) cannot carry a protective order — it rounds below the
# exchange's minimum size — and is not exposure worth managing. ONE rule for every view of "what do we
# hold": the positions the model sees (agent.reconcile_spot_positions), holdings discovery
# (main._discover_unlisted_holdings) and whether a coin may leave the universe (tools.remove_coin).
# 2026-09-25/26: a ~$0.005 KCS remainder was shown to the model as a position (it tried TP/SL on it) and
# blocked KCS's removal from the watchlist.
SPOT_DUST_VALUE_USD = 0.50


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
