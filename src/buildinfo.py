"""Which build made a call — stamped on every signal probe, gate refusal and entry (report-only).

2026-09-28: the rally-week continuation evidence (t~2) came from gpt-5.6-luna; the deploy of Sep 23 20:02
switched the model to gpt-6-luna, and the Sep 25 prompt change ("state the confidence you hold") moved
stated confidence from ~0.76 to ~0.69 on the SAME model. The verdict that then benched continuation rested
only on post-switch calls, and nothing on any row said which code or prompt produced it, so the model switch
and the prompt change could not be separated. ``entryContext.model`` names the deployment; this adds the
code build and the exact trading prompt the call was made under.

Nothing reads these stamps at decision time. Every function here is total: it returns None / {} rather than
raising into a trade.
"""
from __future__ import annotations

import hashlib
import logging
import re
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_REPO = Path(__file__).resolve().parents[1]
_SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")


def _git(*args: str) -> Optional[str]:
  try:
    out = subprocess.run(
      ["git", "-C", str(_REPO), *args], capture_output=True, text=True, timeout=5, check=False,
    )
  except Exception:
    return None
  if out.returncode != 0:
    return None
  return out.stdout


def _head_from_files() -> Optional[str]:
  """HEAD's commit read straight from .git — for a VM whose service user cannot run git on the checkout
  (git's 'dubious ownership' refusal) or has no git binary at all."""
  try:
    git_dir = _REPO / ".git"
    head = (git_dir / "HEAD").read_text().strip()
    if head.startswith("ref:"):
      ref = head.split(":", 1)[1].strip()
      ref_file = git_dir / ref
      if ref_file.is_file():
        return ref_file.read_text().strip()
      packed = git_dir / "packed-refs"
      if packed.is_file():
        for line in packed.read_text().splitlines():
          parts = line.strip().split(" ", 1)
          if len(parts) == 2 and parts[1] == ref:
            return parts[0]
      return None
    return head
  except Exception:
    return None


@lru_cache(maxsize=1)
def code_build() -> Optional[str]:
  """Short commit id of the running checkout ('+dirty' when tracked files differ from it), or None.

  Cached for the process: code and config are read at startup (a change needs a restart to deploy), so one
  process is one build. Never raises.
  """
  sha = (_git("rev-parse", "HEAD") or "").strip() or (_head_from_files() or "")
  sha = sha.strip().lower()
  if not _SHA_RE.match(sha):
    return None
  out = sha[:10]
  status = _git("status", "--porcelain", "--untracked-files=no")
  if status is not None and status.strip():
    out += "+dirty"
  return out


def prompt_hash(text: Any) -> Optional[str]:
  """First 12 hex chars of sha256 over the exact prompt text, or None for an empty/unusable one."""
  try:
    if not text:
      return None
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:12]
  except Exception:
    return None


def build_stamp(prompt_text: Any = None) -> Dict[str, str]:
  """``{'code': <commit>, 'prompt': <hash>}`` with whichever parts are knowable ({} if neither)."""
  out: Dict[str, str] = {}
  try:
    code = code_build()
    if code:
      out["code"] = code
    p = prompt_hash(prompt_text)
    if p:
      out["prompt"] = p
  except Exception as exc:
    logger.debug("build stamp unavailable: %s", exc)
  return out


_CODE_RE = re.compile(r"^[0-9a-f]{7,40}(\+dirty)?$")
_PROMPT_RE = re.compile(r"^[0-9a-f]{8,64}$")


def sanitize_build(value: Any) -> Optional[Dict[str, str]]:
  """Whitelist a stamp for storage: only ``code`` / ``prompt`` in their exact shapes, else None."""
  if not isinstance(value, dict):
    return None
  out: Dict[str, str] = {}
  code = str(value.get("code") or "").strip().lower()
  if _CODE_RE.match(code):
    out["code"] = code
  prompt = str(value.get("prompt") or "").strip().lower()
  if _PROMPT_RE.match(prompt):
    out["prompt"] = prompt
  return out or None
