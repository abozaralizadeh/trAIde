"""The public taker tape (src/kucoin.py KucoinFuturesClient.get_trade_history)."""

import pytest


class TestFuturesTakerTape:
  """The public trade tape is the ONLY KuCoin endpoint carrying the aggressor's side.

  Futures klines return `[ts, o, c, h, l, volume, turnover]` — no taker split — so if this call
  degrades, the flow measurement silently has nothing to measure.
  """

  @staticmethod
  def _client(response):
    from src.kucoin import KucoinFuturesClient
    client = KucoinFuturesClient.__new__(KucoinFuturesClient)   # no creds: the endpoint is public
    calls = []
    def _request(method, path, query=None, **kwargs):
      calls.append((method, path, query))
      if isinstance(response, Exception):
        raise response
      return response
    client._request = _request
    return client, calls

  def test_reads_the_public_tape_for_the_requested_contract(self):
    rows = [{"sequence": 2, "side": "buy", "size": 3, "price": "1", "ts": 2},
            {"sequence": 1, "side": "sell", "size": 1, "price": "1", "ts": 1}]
    client, calls = self._client(rows)
    assert client.get_trade_history("XBTUSDTM") == rows
    assert calls == [("GET", "/api/v1/trade/history", {"symbol": "XBTUSDTM"})]

  @pytest.mark.parametrize("payload", [None, {}, {"data": []}, "", 0])
  def test_a_non_list_payload_reads_as_an_empty_tape(self, payload):
    """KuCoin answers 200000 with a null `data` for an unknown or delisted contract; the caller
    must get an empty tape, not something that explodes when iterated."""
    client, _ = self._client(payload)
    assert client.get_trade_history("NOPE-USDTM") == []
