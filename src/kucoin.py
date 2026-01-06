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


@dataclass
class KucoinOrderResponse:
  orderId: str
  clientOid: str


class KucoinClient:
  def __init__(self, cfg: AppConfig) -> None:
    self.api_key = cfg.kucoin.api_key
    self.api_secret = cfg.kucoin.secret
    self.api_passphrase = cfg.kucoin.passphrase
    self.base_url = cfg.kucoin.base_url.rstrip("/")

  def _sign_headers(self, method: RestMethod, path: str, body: Any | None) -> Dict[str, str]:
    timestamp = str(int(time.time() * 1000))
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
      raise RuntimeError(f"Kucoin HTTP {response.status_code}: {response.text}")

    payload = response.json()
    if payload.get("code") != "200000":
      raise RuntimeError(f"Kucoin API error: {payload.get('code')} {payload.get('msg','')}")

    return payload.get("data")

  def get_trade_accounts(self) -> list[KucoinAccount]:
    data = self._request(
      "GET",
      "/api/v1/accounts",
      auth=True,
      query={"type": "trade"},
    )
    return [KucoinAccount(**item) for item in data]

  def get_ticker(self, symbol: str) -> KucoinTicker:
    data = self._request(
      "GET",
      "/api/v1/market/orderbook/level1",
      query={"symbol": symbol},
    )
    return KucoinTicker(**data)

  def place_order(self, order: KucoinOrderRequest) -> KucoinOrderResponse:
    payload = {k: v for k, v in order.__dict__.items() if v is not None}
    data = self._request(
      "POST",
      "/api/v1/orders",
      auth=True,
      body=payload,
    )
    return KucoinOrderResponse(**data)
