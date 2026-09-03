"""Every @function_tool must compile to a STRICT JSON schema.

Origin (2026-09-03): `log_macro_calendar(events: List[Dict[str, Any]])` was shipped and the bot ran
for ~16 hours doing nothing. A free-form dict emits `additionalProperties` in the generated schema,
which the Agents SDK's strict mode rejects — and it raises inside `build_tools`, so the failure is not
scoped to that one tool: EVERY trading-agent run died at startup with

    agents.exceptions.UserError: additionalProperties should not be set for object types

The whole suite passed throughout, because nothing here had ever called `build_tools`. That is the
real gap this file closes: tool *signatures* are production surface, and they are only validated when
the schema is actually generated. Structured parameters must use an explicit pydantic model with
`extra="forbid"` and no optional fields (strict schemas allow neither).
"""
from types import SimpleNamespace

import pytest

from src.config import load_config
from src.memory import MemoryStore
from src.tools import build_tools


def _ctx(tmp_path) -> SimpleNamespace:
  """The same context agent.py passes, with inert stand-ins — no network, no exchange."""
  return SimpleNamespace(
    cfg=load_config(),
    kucoin=None,
    kucoin_futures=None,
    memory=MemoryStore(str(tmp_path / "mem.json")),
    snapshot=SimpleNamespace(tickers={}, futures_positions=[], futures_pending_orders=[],
                             spot_pending_orders=[], balances={}),
    allowed_symbols=["BTC-USDT"],
    balances_by_currency={},
    fees={"futures_taker": 0.0006},
    _daily_gate_state={},
    _futures_margin_mode={},
    _apply_cross_leverage=lambda *a, **k: None,
    _btc_daily_bias=lambda *a, **k: "neutral",
    _edge_state=lambda *a, **k: {},
    _fee_adjusted_breakeven=lambda *a, **k: 0.0,
    _get_contract_spec=lambda *a, **k: {},
    _repair_allowed_symbol=lambda s, *a, **k: s,
    _spot_position_info=lambda *a, **k: {},
    _spot_position_size=lambda *a, **k: 0.0,
    _stop_distance_ok=lambda *a, **k: True,
    safety_state=SimpleNamespace(),
    entry_token=None,
  )


def test_build_tools_compiles_every_tool_schema(tmp_path):
  """The regression guard. `@function_tool` generates and validates each schema at decoration time,
  so simply building the tools is the assertion — this raised UserError for 16 hours in production
  while every other test passed."""
  tools = build_tools(_ctx(tmp_path))
  assert tools.log_macro_calendar is not None
  assert tools.place_futures_limit_order is not None


def test_structured_tool_params_produce_a_strict_schema(tmp_path):
  """Directly assert the property that broke: no `additionalProperties: true` anywhere in the params
  of any generated tool, which is what a bare Dict[str, Any] parameter produces."""
  tools = build_tools(_ctx(tmp_path))
  checked = 0
  for name in dir(tools):
    if name.startswith("__"):
      continue
    tool = getattr(tools, name)
    schema = getattr(tool, "params_json_schema", None)
    if not isinstance(schema, dict):
      continue
    checked += 1

    def walk(node, path="params"):
      if isinstance(node, dict):
        if node.get("type") == "object":
          assert node.get("additionalProperties") is not True, f"{name} at {path}"
        for k, v in node.items():
          walk(v, f"{path}.{k}")
      elif isinstance(node, list):
        for i, v in enumerate(node):
          walk(v, f"{path}[{i}]")

    walk(schema)
  assert checked > 10, f"expected to inspect many tool schemas, only saw {checked}"


def test_macro_calendar_model_round_trips_into_the_store(tmp_path):
  """The declared model must carry the Research Agent's payload into memory unchanged. Invoking
  through the SDK needs a live ToolContext, so this exercises the model -> store path the tool body
  uses, which is where the shape could actually drift."""
  import time
  from src.tools import MacroEventInput
  m = MemoryStore(str(tmp_path / "mem.json"))
  events = [
    MacroEventInput(name="US CPI (Aug)", ts=int(time.time() + 3600), impact="high"),
    MacroEventInput(name="FOMC statement", ts=int(time.time() + 3 * 86400), impact="high"),
  ]
  assert m.record_macro_events([e.model_dump() for e in events]) == 2
  assert [e["name"] for e in m.macro_events()] == ["US CPI (Aug)", "FOMC statement"]


def test_macro_event_model_rejects_extra_and_missing_fields():
  """Strict schemas forbid both, so the model must too — otherwise the schema and the runtime
  validation disagree and the mismatch only shows up in production."""
  import time
  from pydantic import ValidationError
  from src.tools import MacroEventInput
  ok = dict(name="US CPI", ts=int(time.time() + 60), impact="high")
  assert MacroEventInput(**ok).name == "US CPI"
  with pytest.raises(ValidationError):
    MacroEventInput(**{**ok, "unexpected": 1})       # extra="forbid"
  for missing in ("name", "ts", "impact"):
    with pytest.raises(ValidationError):
      MacroEventInput(**{k: v for k, v in ok.items() if k != missing})
