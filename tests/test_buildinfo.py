"""Build stamp on every call (report-only): which code commit and which exact trading prompt made it.

2026-09-28: the gpt-5.6 -> gpt-6 model switch (Sep 23) and the confidence prompt change (Sep 25) could not be
separated, because no probe said which build produced it.
"""
import re

from src import buildinfo
from src.memory import MemoryStore


def test_code_build_is_a_commit_or_none():
  code = buildinfo.code_build()
  assert code is None or re.match(r"^[0-9a-f]{10}(\+dirty)?$", code), code


def test_prompt_hash_is_deterministic_and_changes_with_the_text():
  assert buildinfo.prompt_hash("abc") == buildinfo.prompt_hash("abc")
  assert buildinfo.prompt_hash("abc") != buildinfo.prompt_hash("abd")
  assert len(buildinfo.prompt_hash("abc")) == 12
  assert buildinfo.prompt_hash("") is None and buildinfo.prompt_hash(None) is None


def test_build_stamp_never_raises(monkeypatch):
  def boom():
    raise RuntimeError("no git")
  monkeypatch.setattr(buildinfo, "code_build", boom)
  assert buildinfo.build_stamp("prompt") == {}


def test_sanitize_build_whitelists_exact_shapes():
  good = {"code": "abcdef1234+dirty", "prompt": "0123456789ab", "secret": "x"}
  assert buildinfo.sanitize_build(good) == {"code": "abcdef1234+dirty", "prompt": "0123456789ab"}
  assert buildinfo.sanitize_build({"code": "not a sha", "prompt": "zz"}) is None
  assert buildinfo.sanitize_build("abcdef1234") is None
  assert buildinfo.sanitize_build(None) is None


def test_head_is_readable_without_the_git_binary(monkeypatch):
  monkeypatch.setattr(buildinfo, "_git", lambda *a: None)
  buildinfo.code_build.cache_clear()
  try:
    head = buildinfo._head_from_files()
    code = buildinfo.code_build()
    if head is None:
      assert code is None
    else:
      assert code == head.strip().lower()[:10]
  finally:
    buildinfo.code_build.cache_clear()


def test_signal_and_gate_probes_store_the_sanitized_stamp(tmp_path):
  mem = MemoryStore(str(tmp_path / "m.json"))
  stamp = {"code": "abcdef1234", "prompt": "0123456789ab", "junk": 1}
  assert mem.record_signal_probe("SPX-USDT", "buy", 1.0, "continuation", build=stamp)
  assert mem.signal_probes(limit=0)[0]["entryContext"]["build"] == {"code": "abcdef1234", "prompt": "0123456789ab"}
  assert mem.record_gate_probe("SPX-USDT", "sell", 1.0, "h1_align", build=stamp)
  assert mem.gate_probes()[0]["entryContext"]["build"] == {"code": "abcdef1234", "prompt": "0123456789ab"}
  # No stamp -> no key (legacy rows stay distinguishable from stamped ones).
  assert mem.record_signal_probe("ETH-USDT", "sell", 1.0, "breakout")
  rows = [r for r in mem.signal_probes(limit=0) if r["entryContext"].get("setupFamily") == "breakout"]
  assert "build" not in rows[0]["entryContext"]
