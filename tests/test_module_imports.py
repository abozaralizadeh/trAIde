"""Every src module must import cleanly — the guard against a partial commit reaching production.

On 2026-08-10 commit d95f709 shipped `src/tools.py` (which imports `fade_setup_available`) without
`src/regime.py` (which defines it). The working tree was fine and the whole suite passed, because
pytest runs against the working tree, not against what was actually committed. The server pulled the
commit and every agent run died with ImportError — the bot could not trade at all until it was fixed.

A test that imports each module by name catches this the moment it is run on the committed checkout
(CI, or a fresh clone), instead of at 2am in the trading loop. It is deliberately dumb: no mocking,
no fixtures, just "does the package hold together".
"""

import importlib
import pkgutil

import pytest

import src

_MODULES = sorted(m.name for m in pkgutil.iter_modules(src.__path__) if not m.name.startswith("_"))


def test_the_package_actually_has_modules_to_check():
    # Guard the guard: if discovery silently returned nothing, the parametrised test below would
    # vacuously pass and this file would provide no protection at all.
    assert len(_MODULES) > 5, _MODULES


@pytest.mark.parametrize("name", _MODULES)
def test_module_imports_cleanly(name):
    """A NameError/ImportError here means the commit is internally inconsistent — a caller was
    shipped without its callee, or a helper was renamed in one file but not another."""
    importlib.import_module(f"src.{name}")
