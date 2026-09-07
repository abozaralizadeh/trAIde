"""Every src module must import cleanly — the guard against a partial commit reaching production.

On 2026-08-10 commit d95f709 shipped `src/tools.py` (which imports `fade_setup_available`) without
`src/regime.py` (which defines it). The working tree was fine and the whole suite passed, because
pytest runs against the working tree, not against what was actually committed. The server pulled the
commit and every agent run died with ImportError — the bot could not trade at all until it was fixed.

A test that imports each module by name catches this the moment it is run on the committed checkout
(CI, or a fresh clone), instead of at 2am in the trading loop. It is deliberately dumb: no mocking,
no fixtures, just "does the package hold together".

Importing only proves the *module level* holds together, though, and on 2026-09-04 that was not
enough: `tools.py` imports `normalize_symbol as _normalize_symbol`, and a helper deep inside
`build_tools` called the unaliased `normalize_symbol`. Import succeeded, the whole suite passed, and
the NameError only fired in the live order path — where it broke atomic entries outright and, at the
probe call site, was swallowed as a failed probe, freezing every edge verdict for three days. So the
second test here reads each module's AST and checks that names used inside function bodies are
actually bound somewhere in the file. Cheap, dependency-free, and it covers every function the unit
tests do not reach.
"""

import ast
import builtins
import importlib
import pkgutil
from pathlib import Path

import pytest

import src

_MODULES = sorted(m.name for m in pkgutil.iter_modules(src.__path__) if not m.name.startswith("_"))

# Names the interpreter provides that never appear as a binding in the source.
_IMPLICIT = {"__file__", "__name__", "__doc__", "__package__", "__spec__", "__loader__", "__debug__"}


def _bound_names(tree: ast.AST) -> set:
    """Every name bound anywhere in the module, regardless of scope.

    Deliberately scope-blind: resolving closures properly would risk false positives on a codebase
    built out of nested factories like `build_tools`, and a false alarm in a test that guards
    production is worse than a slightly coarse one. Anything bound *nowhere* in the file is a typo or
    a rename that missed a caller — which is the whole failure mode being guarded.
    """
    bound = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            bound.update((a.asname or a.name.split(".")[0]) for a in node.names)
        elif isinstance(node, ast.arguments):
            bound.update(a.arg for a in (*node.posonlyargs, *node.args, *node.kwonlyargs))
            bound.update(a.arg for a in (node.vararg, node.kwarg) if a)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            bound.update(node.names)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            bound.add(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            bound.add(node.rest)
    return bound


def test_the_package_actually_has_modules_to_check():
    # Guard the guard: if discovery silently returned nothing, the parametrised test below would
    # vacuously pass and this file would provide no protection at all.
    assert len(_MODULES) > 5, _MODULES


@pytest.mark.parametrize("name", _MODULES)
def test_module_imports_cleanly(name):
    """A NameError/ImportError here means the commit is internally inconsistent — a caller was
    shipped without its callee, or a helper was renamed in one file but not another."""
    importlib.import_module(f"src.{name}")


@pytest.mark.parametrize("name", _MODULES)
def test_no_name_is_used_that_is_never_defined(name):
    """Catch the typo that import alone cannot see: a name used inside a function body that exists
    nowhere in the file. This is what shipped `normalize_symbol` into the live entry path."""
    path = Path(src.__path__[0]) / f"{name}.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    known = _bound_names(tree) | set(dir(builtins)) | _IMPLICIT
    undefined = sorted({
        node.id for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id not in known
    })
    assert not undefined, f"src/{name}.py uses names that are defined nowhere in the file: {undefined}"
