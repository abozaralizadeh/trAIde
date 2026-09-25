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

# The six variables `config.validate_config` requires, forced to dummies BEFORE any `src` import
# (2026-09-25 review). Many harnesses call `load_config()`, so on a clean checkout / fresh VM with no
# `.env` 68 tests failed on 'Missing required configuration', and on the dev machine every test cfg
# carried the REAL KuCoin/Azure keys. Assigned (not setdefault): `load_dotenv()` never overrides a
# variable that is already set, so the real secrets never enter the test process. Tests use fakes only;
# nothing may reach the network with these anyway.
TEST_REQUIRED_ENV = {
  "AZURE_OPENAI_ENDPOINT": "https://test.invalid",
  "AZURE_OPENAI_API_KEY": "test",
  "KUCOIN_API_KEY": "test",
  "KUCOIN_API_SECRET": "test",
  "KUCOIN_API_PASSPHRASE": "test",
  "COINS": "BTC-USDT",
}
os.environ.update(TEST_REQUIRED_ENV)
