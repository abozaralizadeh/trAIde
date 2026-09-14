"""One backend without the deployment must not take the whole bot dark.

Live, 2026-09-14: the endpoint is an APIM load balancer (`.../abopenailb`) over several Azure OpenAI
backends. One backend lost the `gpt-5.6-luna` deployment and the tape shows the consequence exactly —
calls alternated 200 / 404 (10 vs 8), and because a single 404 aborts an agent run, EVERY run from
15:10 onward failed. The bot stopped trading entirely on a fault affecting half of one pool.

The OpenAI SDK retries 408/409/429/5xx and connection errors. A 404 is a client error and treated as
permanent — correct for one endpoint, wrong for a pool.
"""
import asyncio
import json

import httpx
import pytest

from src.agent import _RetryDeploymentMissTransport, _DEPLOYMENT_MISS_RETRIES

DEPLOY_MISS = json.dumps({"error": {
  "message": "Could not find an existing deployment to match the model in the request. "
             "Please verify the model matches an existing deployment in the account.",
  "type": "user_error"}}).encode()


class _ScriptedTransport(httpx.AsyncBaseTransport):
  """Replays a fixed sequence of responses, counting dispatches."""
  def __init__(self, script):
    self.script = list(script)
    self.calls = 0
  async def handle_async_request(self, request):
    self.calls += 1
    status, body = self.script[min(self.calls - 1, len(self.script) - 1)]
    return httpx.Response(status, content=body, request=request)


def _run(script, attempts=_DEPLOYMENT_MISS_RETRIES):
  inner = _ScriptedTransport(script)
  t = _RetryDeploymentMissTransport(inner, attempts=attempts)
  req = httpx.Request("POST", "https://pocs-abozar-apim.azure-api.net/abopenailb/openai/responses")
  return asyncio.run(t.handle_async_request(req)), inner


def test_a_deployment_miss_retries_onto_a_healthy_backend():
  """The live pattern: first dispatch hits the bad backend, the retry lands on a good one."""
  resp, inner = _run([(404, DEPLOY_MISS), (200, b'{"ok":true}')])
  assert resp.status_code == 200
  assert inner.calls == 2


def test_it_gives_up_and_surfaces_the_error_when_every_backend_misses():
  """A deployment absent EVERYWHERE is a config error and must stay loud, not retry forever."""
  resp, inner = _run([(404, DEPLOY_MISS)])
  assert resp.status_code == 404
  assert b"Could not find an existing deployment" in resp.content
  assert inner.calls == _DEPLOYMENT_MISS_RETRIES


def test_an_unrelated_404_is_not_retried():
  """A bad route is genuinely permanent — retrying it just burns calls."""
  resp, inner = _run([(404, b'{"error":{"message":"Resource not found"}}')])
  assert resp.status_code == 404
  assert inner.calls == 1


@pytest.mark.parametrize("status", [200, 400, 429, 500, 503])
def test_every_other_status_passes_straight_through(status):
  """The SDK owns retry policy for these; this transport must not double-retry them."""
  resp, inner = _run([(status, b'{}')])
  assert resp.status_code == status
  assert inner.calls == 1


def test_the_response_body_survives_the_retry_machinery():
  """The body is read to inspect it, so it must be rebuilt or the caller gets an empty stream."""
  resp, _ = _run([(404, DEPLOY_MISS), (200, b'{"id":"resp_123","output":[]}')])
  assert json.loads(resp.content)["id"] == "resp_123"


def test_attempts_are_bounded_even_with_a_pathological_setting():
  resp, inner = _run([(404, DEPLOY_MISS)], attempts=0)
  assert inner.calls == 1 and resp.status_code == 404
