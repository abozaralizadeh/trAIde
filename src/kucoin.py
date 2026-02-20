from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

import requests

from .config import AppConfig

RestMethod = Literal["GET", "POST", "DELETE"]


@dataclass
class KucoinAccount:
  id: str
  currency: str
  type: str
  balance: str
  available: str
  holds: str


@dataclass
class KucoinTicker:
  sequence: str
  bestAsk: str
  size: str
  price: str
  bestBidSize: str
  bestBid: str
  bestAskSize: str
  time: int


@dataclass
class KucoinOrderRequest:
  symbol: str
  side: Literal["buy", "sell"]
  type: Literal["market", "limit"]
  size: Optional[str] = None
  funds: Optional[str] = None
  price: Optional[str] = None
  clientOid: Optional[str] = None
  stopPrice: Optional[str] = None
  stopPriceType: Optional[Literal["TP", "MP"]] = None  # TP=trigger when price rises, MP=trigger when price falls


@dataclass
class KucoinOrderResponse:
  orderId: str
  clientOid: Optional[str] = None


@dataclass
class KucoinFuturesOrderRequest:
  symbol: str
  side: Literal["buy", "sell"]
  type: Literal["market", "limit"]
  leverage: str
  size: str
  price: Optional[str] = None
  clientOid: Optional[str] = None
  stop: Optional[Literal["up", "down"]] = None
  stopPriceType: Optional[Literal["TP", "MP", "IP"]] = None  # TP=last price, MP=mark, IP=index (per API)
  stopPrice: Optional[str] = None
  reduceOnly: Optional[bool] = None
  closeOrder: Optional[bool] = None
  takeProfitPrice: Optional[str] = None
  stopLossPrice: Optional[str] = None
  postOnly: Optional[bool] = None
  hidden: Optional[bool] = None
  iceberg: Optional[bool] = None
  visibleSize: Optional[str] = None
  marginMode: Optional[Literal["cross", "isolated", "CROSS", "ISOLATED"]] = None
  autoDeposit: Optional[bool] = None  # required when using isolated to specify auto margin replenishment


@dataclass
class KucoinFuturesOrderResponse:
  orderId: str
  clientOid: Optional[str] = None


class KucoinClient:
  def __init__(self, cfg: AppConfig) -> None:
    self.api_key = cfg.kucoin.api_key
    self.api_secret = cfg.kucoin.secret
    self.api_passphrase = cfg.kucoin.passphrase
    self.base_url = cfg.kucoin.base_url.rstrip("/")
    self._time_offset_ms = 0

  def _timestamp_ms(self) -> int:
    return int(time.time() * 1000 + self._time_offset_ms)

  def _sync_time(self) -> None:
    try:
      resp = requests.get(f"{self.base_url}/api/v1/timestamp", timeout=5)
      if resp.ok:
        server_ms = resp.json().get("data")
        if isinstance(server_ms, (int, float)):
          local_ms = int(time.time() * 1000)
          self._time_offset_ms = int(server_ms) - local_ms
    except Exception:
      pass

  def _sign_headers(self, method: RestMethod, path: str, body: Any | None) -> Dict[str, str]:
    timestamp = str(self._timestamp_ms())
    body_str = json.dumps(body) if body is not None else ""
    prehash = f"{timestamp}{method}{path}{body_str}"

    signature = hmac.new(
      self.api_secret.encode("utf-8"),
      prehash.encode("utf-8"),
      hashlib.sha256,
    ).digest()
    passphrase = hmac.new(
      self.api_secret.encode("utf-8"),
      self.api_passphrase.encode("utf-8"),
      hashlib.sha256,
    ).digest()

    return {
      "KC-API-KEY": self.api_key,
      "KC-API-SIGN": base64.b64encode(signature).decode("utf-8"),
      "KC-API-TIMESTAMP": timestamp,
      "KC-API-PASSPHRASE": base64.b64encode(passphrase).decode("utf-8"),
      "KC-API-KEY-VERSION": "2",
    }

  def _request(
    self,
    method: RestMethod,
    path: str,
    *,
    auth: bool = False,
    query: Optional[Dict[str, str | int | float]] = None,
    body: Any | None = None,
  ) -> Any:
    qs = f"?{requests.compat.urlencode(query)}" if query else ""
    full_path = f"{path}{qs}"
    url = f"{self.base_url}{full_path}"

    headers = {"Content-Type": "application/json"}
    if auth:
      headers.update(self._sign_headers(method, full_path, body))

    response = requests.request(method, url, headers=headers, json=body, timeout=15)
    if not response.ok:
      # Retry once on timestamp errors after syncing time.
      if response.status_code == 400 and "Invalid KC-API-TIMESTAMP" in response.text:
        self._sync_time()
        headers = {"Content-Type": "application/json"}
        if auth:
          headers.update(self._sign_headers(method, full_path, body))
        response = requests.request(method, url, headers=headers, json=body, timeout=15)
      if not response.ok:
        raise RuntimeError(f"Kucoin HTTP {response.status_code}: {response.text}")

    payload = response.json()
    if payload.get("code") != "200000":
      raise RuntimeError(f"Kucoin API error: {payload.get('code')} {payload.get('msg','')}")

    return payload.get("data")

  def get_trade_accounts(self) -> list[KucoinAccount]:
    return self.get_accounts("trade")

  def get_financial_accounts(self) -> list[KucoinAccount]:
    # KuCoin Earn (Financial/Pool-X) balances are not returned by /api/v1/accounts.
    # Try Earn holdings endpoints first, then fall back to the legacy pool account type.
    for path in ("/api/v1/earn/hold-assets", "/api/v1/earn/holdings", "/api/v1/earn/holding"):
      try:
        data = self._request("GET", path, auth=True)
        accounts = self._map_earn_holdings(data)
        if accounts:
          return accounts
      except Exception:
        continue
    return self.get_accounts("pool")

  def _map_earn_holdings(self, data: Any) -> list[KucoinAccount]:
    items: Any = data
    if isinstance(data, dict):
      for key in ("items", "data", "list", "holdings"):
        if isinstance(data.get(key), list):
          items = data.get(key)
          break
    if not isinstance(items, list):
      return []

    accounts: list[KucoinAccount] = []
    for idx, item in enumerate(items):
      if not isinstance(item, dict):
        continue
      currency = (
        item.get("currency")
        or item.get("coin")
        or item.get("asset")
        or item.get("symbol")
      )
      if not currency:
        continue

      balance = (
        item.get("holdAmount")
        or item.get("holding")
        or item.get("amount")
        or item.get("totalAmount")
        or item.get("total")
        or item.get("principal")
        or item.get("balance")
      )
      available = (
        item.get("redeemableAmount")
        or item.get("available")
        or item.get("redeemable")
        or balance
      )
      holds = item.get("locked") or item.get("hold") or "0"

      accounts.append(
        KucoinAccount(
          id=str(item.get("id") or f"earn:{currency}:{idx}"),
          currency=str(currency),
          type="financial",
          balance=str(balance or "0"),
          available=str(available or "0"),
          holds=str(holds or "0"),
        )
      )
    return accounts

  def get_accounts(self, account_type: Optional[str] = None) -> list[KucoinAccount]:
    """Fetch accounts; when account_type is None, returns all (main/trade/margin)."""
    query = {"type": account_type} if account_type else None
    data = self._request(
      "GET",
      "/api/v1/accounts",
      auth=True,
      query=query,
    )
    return [KucoinAccount(**item) for item in data]

  def get_ticker(self, symbol: str) -> KucoinTicker:
    data = self._request(
      "GET",
      "/api/v1/market/orderbook/level1",
      query={"symbol": symbol},
    )
    if not isinstance(data, dict):
      raise RuntimeError(f"Kucoin ticker payload missing for {symbol}: {data}")
    return KucoinTicker(**data)

  def get_candles(
    self,
    symbol: str,
    *,
    interval: str = "1min",
    start_at: Optional[int] = None,
    end_at: Optional[int] = None,
  ) -> list[list[str]]:
    """Fetch recent candles. Interval examples: 1min, 5min, 15min, 1hour."""
    query: Dict[str, str | int] = {"symbol": symbol, "type": interval}
    if start_at is not None:
      query["startAt"] = start_at
    if end_at is not None:
      query["endAt"] = end_at

    data = self._request(
      "GET",
      "/api/v1/market/candles",
      query=query,
    )
    # API returns list of [time, open, close, high, low, volume, turnover].
    return data

  def get_orderbook_levels(self, symbol: str, depth: int = 20) -> Dict[str, Any]:
    """Fetch level2 orderbook snapshot. Depth must be 20 or 100."""
    depth_allowed = 20 if depth <= 20 else 100
    path = "/api/v1/market/orderbook/level2_20" if depth_allowed == 20 else "/api/v1/market/orderbook/level2_100"
    return self._request(
      "GET",
      path,
      query={"symbol": symbol},
    )

  def transfer_funds(
    self,
    *,
    currency: str,
    amount: float,
    from_account: str,
    to_account: str,
    client_oid: Optional[str] = None,
  ) -> Dict[str, Any]:
    """Transfer funds between internal accounts (spot/futures/unified) with universal transfer, falling back if needed."""
    def _normalize_universal(account: str) -> str:
      acct = (account or "").lower()
      mapping = {
        "trade": "TRADE",
        "spot": "TRADE",
        "main": "MAIN",
        "funding": "MAIN",
        "margin": "MARGIN",
        "contract": "CONTRACT",
        "futures": "CONTRACT",
        "financial": "POOLED",
        "pool": "POOLED",
        "pool-x": "POOLED",
        "unified": "UNIFIED",
      }
      return mapping.get(acct, acct.upper())

    def _normalize_legacy(account: str) -> str:
      acct = (account or "").lower()
      mapping = {
        "trade": "trade",
        "spot": "trade",
        "main": "main",
        "funding": "main",
        "margin": "margin",
        "financial": "pool",
        "pool": "pool",
        "pool-x": "pool",
        "contract": "contract",
        "futures": "contract",
      }
      return mapping.get(acct, acct)

    base_oid = client_oid or str(int(time.time() * 1000))
    cur = currency.upper()
    amt = f"{amount}"
    from_u = _normalize_universal(from_account)
    to_u = _normalize_universal(to_account)
    from_l = _normalize_legacy(from_account)
    to_l = _normalize_legacy(to_account)

    def _universal(from_type: str, to_type: str, oid: str) -> Dict[str, Any]:
      payload = {
        "clientOid": oid,
        "currency": cur,
        "amount": amt,
        "type": "INTERNAL",
        "fromAccountType": from_type,
        "toAccountType": to_type,
      }
      return self._request(
        "POST",
        "/api/v3/accounts/universal-transfer",
        auth=True,
        body=payload,
      )

    def _inner(from_type: str, to_type: str, oid: str) -> Dict[str, Any]:
      legacy_body = {
        "clientOid": oid,
        "currency": cur,
        "from": from_type,
        "to": to_type,
        "amount": amt,
      }
      return self._request(
        "POST",
        "/api/v2/accounts/inner-transfer",
        auth=True,
        body=legacy_body,
      )

    # Financial (POOLED/POOL) is not accepted by universal transfer on many keys.
    if "POOLED" in (from_u, to_u):
      # Bridge financial <-> futures through spot (trade), since inner-transfer does not support contract.
      if from_u == "POOLED" and to_u == "CONTRACT":
        step1 = _inner("pool", "trade", f"{base_oid}-1")
        step2 = _universal("TRADE", "CONTRACT", f"{base_oid}-2")
        return {"bridge": "financial->trade->futures", "steps": [step1, step2]}
      if from_u == "CONTRACT" and to_u == "POOLED":
        step1 = _universal("CONTRACT", "TRADE", f"{base_oid}-1")
        step2 = _inner("trade", "pool", f"{base_oid}-2")
        return {"bridge": "futures->trade->financial", "steps": [step1, step2]}

      # Otherwise prefer inner-transfer for pool/main/trade/margin.
      return _inner(from_l, to_l, base_oid)

    # Default path: try universal transfer, then fallback to inner-transfer when possible.
    try:
      return _universal(from_u, to_u, base_oid)
    except Exception as exc:
      if from_l == "contract" or to_l == "contract":
        # Inner transfer does not support contract accounts; re-raise original error.
        raise exc
      try:
        return _inner(from_l, to_l, base_oid)
      except Exception:
        # Re-raise original failure for easier debugging if fallback also fails.
        raise exc

  def place_order(self, order: KucoinOrderRequest) -> KucoinOrderResponse:
    payload = {k: v for k, v in order.__dict__.items() if v is not None}
    data = self._request(
      "POST",
      "/api/v1/orders",
      auth=True,
      body=payload,
    )
    order_id = data.get("orderId") if isinstance(data, dict) else None
    client_oid = data.get("clientOid") if isinstance(data, dict) else None
    return KucoinOrderResponse(orderId=order_id or "", clientOid=client_oid or payload.get("clientOid"))

  def get_base_fee(self) -> Dict[str, Any]:
    """Fetch base fee (maker/taker) for spot."""
    return self._request(
      "GET",
      "/api/v1/base-fee",
      auth=True,
    ) or {}

  def place_stop_order(self, order: KucoinOrderRequest) -> KucoinOrderResponse:
    """Place a spot stop order (KuCoin stop-order endpoint)."""
    payload = {k: v for k, v in order.__dict__.items() if v is not None}
    data = self._request(
      "POST",
      "/api/v1/stop-order",
      auth=True,
      body=payload,
    )
    order_id = data.get("orderId") if isinstance(data, dict) else None
    client_oid = data.get("clientOid") if isinstance(data, dict) else None
    return KucoinOrderResponse(orderId=order_id or "", clientOid=client_oid or payload.get("clientOid"))

  def cancel_stop_order(self, order_id: Optional[str] = None, client_oid: Optional[str] = None) -> Dict[str, Any]:
    """Cancel a spot stop order by orderId or clientOid."""
    query: Dict[str, str] = {}
    if order_id:
      query["orderId"] = order_id
    if client_oid:
      query["clientOid"] = client_oid
    try:
      return self._request(
        "DELETE",
        "/api/v1/stop-order",
        auth=True,
        query=query or None,
      )
    except Exception as exc:
      # Fallback to explicit cancel endpoint (some accounts require this path)
      try:
        return self._request(
          "DELETE",
          "/api/v1/stop-order/cancel",
          auth=True,
          query=query or None,
        )
      except Exception:
        raise exc

  def list_stop_orders(self, status: str = "active", symbol: Optional[str] = None) -> list[Dict[str, Any]]:
    """List spot stop orders (default: active)."""
    query: Dict[str, str] = {"status": status}
    if symbol:
      query["symbol"] = symbol
    data = self._request(
      "GET",
      "/api/v1/stop-order",
      auth=True,
      query=query,
    )
    # API may return {"items":[...]} or list; normalize
    if isinstance(data, dict) and "items" in data:
      return data.get("items") or []
    return data or []


class KucoinFuturesClient:
  def __init__(self, cfg: AppConfig) -> None:
    # Futures use the same credentials as spot; only the base URL and enable flag differ.
    self.api_key = cfg.kucoin.api_key
    self.api_secret = cfg.kucoin.secret.encode("utf-8")
    self.api_passphrase = cfg.kucoin.passphrase.encode("utf-8")
    self.base_url = cfg.kucoin_futures.base_url.rstrip("/")
    self._time_offset_ms = 0

  def _timestamp_ms(self) -> int:
    return int(time.time() * 1000 + self._time_offset_ms)

  def _sign_headers(self, method: RestMethod, path: str, body: Any | None) -> Dict[str, str]:
    timestamp = str(self._timestamp_ms())
    body_str = json.dumps(body) if body is not None else ""
    prehash = f"{timestamp}{method}{path}{body_str}"

    signature = hmac.new(self.api_secret, prehash.encode("utf-8"), hashlib.sha256).digest()
    passphrase = hmac.new(self.api_secret, self.api_passphrase, hashlib.sha256).digest()

    return {
      "KC-API-KEY": self.api_key,
      "KC-API-SIGN": base64.b64encode(signature).decode("utf-8"),
      "KC-API-TIMESTAMP": timestamp,
      "KC-API-PASSPHRASE": base64.b64encode(passphrase).decode("utf-8"),
      "KC-API-KEY-VERSION": "2",
    }

  def _request(
    self,
    method: RestMethod,
    path: str,
    *,
    auth: bool = False,
    query: Optional[Dict[str, str | int | float]] = None,
    body: Any | None = None,
  ) -> Any:
    qs = f"?{requests.compat.urlencode(query)}" if query else ""
    full_path = f"{path}{qs}"
    url = f"{self.base_url}{full_path}"

    headers = {"Content-Type": "application/json"}
    if auth:
      headers.update(self._sign_headers(method, full_path, body))

    response = requests.request(method, url, headers=headers, json=body, timeout=15)
    if not response.ok:
      if response.status_code == 400 and "Invalid KC-API-TIMESTAMP" in response.text:
        self._sync_time()
        headers = {"Content-Type": "application/json"}
        if auth:
          headers.update(self._sign_headers(method, full_path, body))
        response = requests.request(method, url, headers=headers, json=body, timeout=15)
      if not response.ok:
        raise RuntimeError(f"Kucoin Futures HTTP {response.status_code}: {response.text}")

    payload = response.json()
    if payload.get("code") not in ("200000", "200"):  # futures may return 200
      raise RuntimeError(f"Kucoin Futures API error: {payload.get('code')} {payload.get('msg','')}")

    return payload.get("data")

  def _sync_time(self) -> None:
    try:
      resp = requests.get(f"{self.base_url}/api/v1/timestamp", timeout=5)
      if resp.ok:
        server_ms = resp.json().get("data")
        if isinstance(server_ms, (int, float)):
          local_ms = int(time.time() * 1000)
          self._time_offset_ms = int(server_ms) - local_ms
    except Exception:
      pass

  def place_order(self, order: KucoinFuturesOrderRequest) -> KucoinFuturesOrderResponse:
    payload = {k: v for k, v in order.__dict__.items() if v is not None}
    data = self._request(
      "POST",
      "/api/v1/orders",
      auth=True,
      body=payload,
    )
    order_id = data.get("orderId") if isinstance(data, dict) else None
    client_oid = data.get("clientOid") if isinstance(data, dict) else None
    return KucoinFuturesOrderResponse(orderId=order_id or "", clientOid=client_oid or payload.get("clientOid", ""))

  def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> Dict[str, Any]:
    """Cancel a futures order (works for stop orders too)."""
    query = {"symbol": symbol} if symbol else None
    return self._request(
      "DELETE",
      f"/api/v1/orders/{order_id}",
      auth=True,
      query=query,
    )

  def list_orders(self, status: str = "open", symbol: Optional[str] = None, side: Optional[str] = None) -> list[Dict[str, Any]]:
    """List futures orders; status may include 'open', 'done', 'active' etc."""
    query: Dict[str, str] = {"status": status}
    if symbol:
      query["symbol"] = symbol
    if side:
      query["side"] = side
    data = self._request(
      "GET",
      "/api/v1/orders",
      auth=True,
      query=query,
    )
    if isinstance(data, dict) and "items" in data:
      return data.get("items") or []
    return data or []

  def list_stop_orders(self, status: str = "active", symbol: Optional[str] = None) -> list[Dict[str, Any]]:
    """List futures stop orders (active or done)."""
    query: Dict[str, str] = {"status": status}
    if symbol:
      query["symbol"] = symbol
    data = self._request(
      "GET",
      "/api/v1/stopOrders",
      auth=True,
      query=query,
    )
    if isinstance(data, dict) and "items" in data:
      return data.get("items") or []
    return data or []

  def get_account_overview(self, currency: str = "USDT") -> Dict[str, Any]:
    """Fetch futures account overview for a given currency (default USDT)."""
    data = self._request(
      "GET",
      "/api/v1/account-overview",
      auth=True,
      query={"currency": currency},
    )
    return data or {}

  def get_position(self, symbol: str) -> Dict[str, Any]:
    """Fetch futures position for a symbol to detect margin mode (cross/isolated)."""
    return self._request(
      "GET",
      "/api/v1/position",
      auth=True,
      query={"symbol": symbol},
    ) or {}

  def list_positions(self, status: Optional[str] = None) -> list[Dict[str, Any]]:
    """List all futures positions (optionally by status)."""
    query: Dict[str, str] = {}
    if status:
      query["status"] = status
    data = self._request(
      "GET",
      "/api/v1/positions",
      auth=True,
      query=query or None,
    )
    if isinstance(data, dict) and "items" in data:
      return data.get("items") or []
    return data or []

  def set_leverage(self, symbol: str, leverage: float, cross: bool = True) -> Dict[str, Any]:
    """Set leverage for a futures symbol using the official leverage endpoint."""
    body = {"symbol": symbol, "leverage": str(int(leverage))}
    return self._request(
      "POST",
      "/api/v1/position/margin/leverage",
      auth=True,
      body=body,
    )

  def set_margin_mode(self, symbol: str, margin_mode: str, auto_deposit: Optional[bool] = None) -> Dict[str, Any]:
    """Set margin mode (CROSS/ISOLATED) for a futures symbol."""
    mode_norm = margin_mode.upper()
    body: Dict[str, Any] = {"symbol": symbol, "marginMode": mode_norm}
    if auto_deposit is not None:
      body["autoDeposit"] = auto_deposit
    return self._request(
      "POST",
      "/api/v1/position/margin/mode",
      auth=True,
      body=body,
    )
