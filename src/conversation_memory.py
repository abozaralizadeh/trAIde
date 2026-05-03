from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)


class ConversationMemory:
  """Conversation history with automatic compaction.

  Keeps the last ``recent_count`` exchanges (user+assistant pairs) verbatim
  and maintains a rolling LLM-generated summary of older messages.
  Thread-safe and JSON-persisted.
  """

  def __init__(
    self,
    file_path: str | Path,
    recent_count: int = 3,
    max_context_message_chars: int = 500,
  ) -> None:
    self.file_path = Path(file_path)
    self.recent_count = recent_count
    self.max_context_message_chars = max_context_message_chars
    self._lock = threading.Lock()

  def _read(self) -> Dict[str, Any]:
    if not self.file_path.exists():
      return {"summary": "", "messages": []}
    try:
      data = json.loads(self.file_path.read_text())
      data.setdefault("summary", "")
      data.setdefault("messages", [])
      return data
    except Exception:
      return {"summary": "", "messages": []}

  def _write(self, data: Dict[str, Any]) -> None:
    self.file_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = self.file_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(self.file_path)

  def add_message(self, role: str, content: str) -> None:
    with self._lock:
      data = self._read()
      data["messages"].append({
        "role": role,
        "content": content,
        "ts": datetime.now(timezone.utc).isoformat(),
      })
      self._write(data)

  def add_exchange(self, user_content: str, assistant_content: str) -> None:
    """Add a user-assistant exchange pair atomically."""
    with self._lock:
      data = self._read()
      now = datetime.now(timezone.utc).isoformat()
      data["messages"].append({"role": "user", "content": user_content, "ts": now})
      data["messages"].append({"role": "assistant", "content": assistant_content, "ts": now})
      self._write(data)

  def get_context(self) -> str:
    """Return formatted context string: summary of old messages + recent exchanges."""
    with self._lock:
      data = self._read()

    messages = data["messages"]
    summary = data["summary"]

    recent_limit = self.recent_count * 2
    recent = messages[-recent_limit:] if len(messages) > recent_limit else messages
    cap = self.max_context_message_chars

    parts: List[str] = []
    if summary:
      parts.append(f"## Previous conversation summary:\n{summary}")

    if recent:
      lines: List[str] = []
      for msg in recent:
        role_label = "User" if msg["role"] == "user" else "Assistant"
        text = msg["content"]
        if len(text) > cap:
          text = text[:cap] + "..."
        lines.append(f"**{role_label}**: {text}")
      parts.append("## Recent messages:\n" + "\n\n".join(lines))

    return "\n\n".join(parts) if parts else ""

  def needs_compaction(self) -> bool:
    with self._lock:
      data = self._read()
    return len(data["messages"]) > self.recent_count * 2

  def compact(self, summarizer: Callable[[str, List[Dict[str, Any]]], str]) -> None:
    """Compact older messages into the rolling summary.

    Args:
      summarizer: callable(existing_summary, old_messages) -> new_summary.
                  If it raises, messages are preserved unchanged.
    """
    with self._lock:
      data = self._read()
      messages = data["messages"]
      recent_limit = self.recent_count * 2

      if len(messages) <= recent_limit:
        return

      old_messages = messages[:-recent_limit]
      recent_messages = messages[-recent_limit:]

      try:
        new_summary = summarizer(data["summary"], old_messages)
      except Exception as exc:
        logger.warning("Conversation compaction failed, keeping old messages: %s", exc)
        return

      data["summary"] = new_summary
      data["messages"] = recent_messages
      self._write(data)

  def clear(self) -> None:
    with self._lock:
      self._write({"summary": "", "messages": []})

  def message_count(self) -> int:
    with self._lock:
      return len(self._read()["messages"])
