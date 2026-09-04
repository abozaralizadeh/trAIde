"""Test-session setup.

The suite must never write to the real trading log. `tests/test_module_imports.py` deliberately
imports EVERY module in `src`, which includes `src.wsgi` — and wsgi attaches a RotatingFileHandler to
the ROOT logger at import time (correctly: gunicorn's serving path is that import). Once attached it
stays for the whole pytest session, so every later test's log output was appended to `traide.log`.

That is not cosmetic. On 2026-09-04 a test run's `[DRY-RUN]` profit-lock warnings landed in the
production log between real entries, and reading them back as production behaviour cost real
debugging time — they look exactly like a live fault ("tickSize lookup failed", "Emergency bracket
PARTIAL") but come from stubs with no exchange client attached.

`LOG_FILE=""` is the existing config knob wsgi already honours (`if cfg.supervisor.log_file:`), so no
production code needs a test-only branch. Set before any test module — and therefore any `src`
import — is collected.
"""
import os

os.environ.setdefault("_TRAIDE_REAL_LOG_FILE", os.environ.get("LOG_FILE", ""))
os.environ["LOG_FILE"] = ""
