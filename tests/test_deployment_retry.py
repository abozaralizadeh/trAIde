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


# --- One client, many event loops -------------------------------------------------------------------
# Live, 2026-09-22: main builds the OpenAI client ONCE, but every agent run is its own `asyncio.run`
# in a worker thread. The retry transport held a single httpx pool, so run #2 found run #1's expired
# keep-alive socket and httpcore closed it on run #1's loop — already closed — raising
# "RuntimeError: Event loop is closed". The first run after the deploy worked; the next 41 all died.

def _loopback_server():
  import threading
  from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

  class _Ok(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # keep-alive, like the real endpoint
    def do_POST(self):
      self.rfile.read(int(self.headers.get("content-length", 0)))
      self.send_response(200)
      self.send_header("content-length", "2")
      self.end_headers()
      self.wfile.write(b"ok")
    def log_message(self, *args):
      pass

  srv = ThreadingHTTPServer(("127.0.0.1", 0), _Ok)
  threading.Thread(target=srv.serve_forever, daemon=True).start()
  return srv, f"http://127.0.0.1:{srv.server_address[1]}/openai/responses"


def _short_keepalive():
  # Expire idle sockets fast so the second run meets an expired connection, as production did
  # after the ~5 min gap between runs.
  return httpx.AsyncHTTPTransport(limits=httpx.Limits(keepalive_expiry=0.05))


def test_one_shared_client_survives_sequential_asyncio_run_loops():
  import threading
  import time
  srv, url = _loopback_server()
  client = httpx.AsyncClient(transport=_RetryDeploymentMissTransport(factory=_short_keepalive))
  results = []
  try:
    for _ in range(3):
      def _one_run():  # same shape as main._run_in_daemon_thread -> agent.asyncio.run
        try:
          results.append(asyncio.run(client.post(url, content=b"{}")).status_code)
        except Exception as exc:  # pragma: no cover - the regression we are guarding
          results.append(f"{type(exc).__name__}: {exc}")
      t = threading.Thread(target=_one_run)
      t.start()
      t.join()
      time.sleep(0.15)
  finally:
    srv.shutdown()
  assert results == [200, 200, 200]


def test_each_loop_gets_its_own_pool_and_dead_loops_are_released():
  made = []
  def _factory():
    made.append(_ScriptedTransport([(200, b"{}")]))
    return made[-1]
  t = _RetryDeploymentMissTransport(factory=_factory)
  req = httpx.Request("POST", "https://example.invalid/openai/responses")

  async def _twice():
    await t.handle_async_request(req)
    await t.handle_async_request(req)
  asyncio.run(_twice())
  assert len(made) == 1  # reused within one loop
  asyncio.run(t.handle_async_request(req))
  assert len(made) == 2  # a new loop never touches the old pool
  assert len(t._per_loop) == 1  # the closed loop's pool was dropped, not leaked
