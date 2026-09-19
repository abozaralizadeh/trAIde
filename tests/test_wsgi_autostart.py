"""`src/wsgi.py` starts the trading loop at module import — that must happen for gunicorn and must
NOT happen for a test runner.

Verified 2026-09-20: `tests/test_module_imports.py` imports every module in `src` to catch partial
commits, and importing `wsgi` spawned `Thread-1 (_start_background_loop)` and `supervisor-bot` — a
LIVE trading loop and the Telegram bot against the real account, on every `pytest` run. A daemon
thread cannot finish an agent run inside a short test, but ProtectionManager places and moves real
orders well within that window.

Both directions are checked in a SUBPROCESS with the trading loop and Telegram bot stubbed out, so the
test can prove the production path still starts without anything real running.
"""
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_HARNESS = """
import sys, types, threading, asyncio, os
# Stub the two things wsgi would actually run, BEFORE importing it.
main_stub = types.ModuleType("src.main")
async def trading_loop():
    await asyncio.sleep(30)
main_stub.trading_loop = trading_loop
sys.modules["src.main"] = main_stub
tg = types.ModuleType("src.telegram_bot")
tg.start_telegram_bot = lambda cfg: None
sys.modules["src.telegram_bot"] = tg
{pytest_line}
before = {{t.name for t in threading.enumerate()}}
import src.wsgi          # noqa: F401  -- the import IS the behaviour under test
started = sorted(t.name for t in threading.enumerate() if t.name not in before)
print("STARTED:" + ",".join(started))
"""


def _run(pretend_pytest: bool, env_extra=None):
  line = 'sys.modules["pytest"] = types.ModuleType("pytest")' if pretend_pytest else ""
  code = textwrap.dedent(_HARNESS).format(pytest_line=line)
  env = {"PATH": "/usr/bin:/bin", "HOME": str(Path.home()), "LOG_FILE": ""}
  env.update(env_extra or {})
  out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                       env=env, timeout=120)
  assert out.returncode == 0, out.stderr[-2000:]
  line = [l for l in out.stdout.splitlines() if l.startswith("STARTED:")]
  assert line, out.stdout + out.stderr
  return [t for t in line[0][len("STARTED:"):].split(",") if t]


def test_production_import_still_starts_the_trading_loop():
  """gunicorn imports this module and never calls `application` until traffic arrives, so the loop
  has to start on import. If this breaks, the bot silently stops trading after a deploy."""
  started = _run(pretend_pytest=False)
  assert any("_start_background_loop" in t for t in started), started


def test_a_test_runner_does_not_start_a_live_trading_loop():
  started = _run(pretend_pytest=True)
  assert started == [], f"importing wsgi under a test runner started: {started}"


def test_the_off_switch_wins_even_outside_a_test_runner():
  assert _run(pretend_pytest=False, env_extra={"TRAIDE_WSGI_AUTOSTART": "0"}) == []


def test_the_on_switch_wins_even_under_a_test_runner():
  started = _run(pretend_pytest=True, env_extra={"TRAIDE_WSGI_AUTOSTART": "1"})
  assert any("_start_background_loop" in t for t in started), started
