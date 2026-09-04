"""fetch_orderbook must not raise when KuCoin returns a null `data` field.

Reported 2026-09-04: `trAIde fetch_orderbook → 'NoneType' object has no attribute 'get'`.

KuCoin answers `code: 200000` with a NULL `data` for an unknown or delisted symbol, and the client
returns `payload["data"]` straight through — so a "successful" call hands back None. The spot tool
then did `ob.get("bids", ...)` on it and the exception propagated OUT of the tool rather than
returning something the model could reason about. The futures sibling survived only because a blanket
try/except swallowed it into an opaque error string; both now share one guarded helper.
"""
import pytest

from src.tools import normalize_orderbook


def test_null_data_returns_an_empty_book_not_an_exception():
  """The exact reported failure: a 200000 response whose data field is null."""
  out = normalize_orderbook(None, 20, "BTC-USDT")
  assert out["orderbook"] == {"bids": [], "asks": []}
  assert "error" in out and "delisted" in out["error"]
  assert out["symbol"] == "BTC-USDT" and out["depth"] == 20


@pytest.mark.parametrize("junk", [None, [], "", 0, "null", ["bids"]])
def test_any_non_dict_payload_degrades_to_an_empty_book(junk):
  out = normalize_orderbook(junk, 20, "X-USDT")
  assert out["orderbook"] == {"bids": [], "asks": []}
  assert "error" in out


@pytest.mark.parametrize("book", [
  {"bids": None, "sequence": "1"},          # side present but null
  {"asks": [], "sequence": "1"},            # side present but empty
  {"sequence": "1"},                        # sides absent entirely
])
def test_a_thin_book_with_missing_sides_is_data_not_a_failure(book):
  """An illiquid pair legitimately has one or both sides empty; that must not read as an error."""
  out = normalize_orderbook(book, 20, "X-USDT")
  assert "error" not in out
  assert out["orderbook"]["bids"] == [] or out["orderbook"]["asks"] == []


def test_a_normal_book_is_trimmed_to_the_requested_depth():
  levels = [[str(i), "1"] for i in range(50)]
  out = normalize_orderbook({"bids": list(levels), "asks": list(levels), "sequence": "9"}, 20, "BTC-USDT")
  assert len(out["orderbook"]["bids"]) == 20
  assert len(out["orderbook"]["asks"]) == 20
  assert out["orderbook"]["sequence"] == "9"      # other fields survive


def test_the_caller_s_payload_is_not_mutated():
  """The tools pass the client's own response object; trimming it in place would corrupt any
  caller that still holds it (and did, before this was a copy)."""
  original = {"bids": [["1", "1"]] * 50, "asks": [["2", "1"]] * 50}
  normalize_orderbook(original, 5, "BTC-USDT")
  assert len(original["bids"]) == 50 and len(original["asks"]) == 50
