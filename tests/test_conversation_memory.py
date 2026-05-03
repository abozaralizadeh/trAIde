import json
from pathlib import Path

import pytest

from src.conversation_memory import ConversationMemory


@pytest.fixture
def mem_file(tmp_path):
  return tmp_path / "conv.json"


class TestAddAndRead:
  def test_add_message_creates_file(self, mem_file):
    cm = ConversationMemory(mem_file, recent_count=3)
    cm.add_message("user", "hello")
    assert mem_file.exists()
    data = json.loads(mem_file.read_text())
    assert len(data["messages"]) == 1
    assert data["messages"][0]["role"] == "user"
    assert data["messages"][0]["content"] == "hello"
    assert "ts" in data["messages"][0]

  def test_add_exchange_adds_pair(self, mem_file):
    cm = ConversationMemory(mem_file, recent_count=3)
    cm.add_exchange("what is BTC?", "BTC is at $67k")
    data = json.loads(mem_file.read_text())
    assert len(data["messages"]) == 2
    assert data["messages"][0]["role"] == "user"
    assert data["messages"][1]["role"] == "assistant"

  def test_message_count(self, mem_file):
    cm = ConversationMemory(mem_file, recent_count=3)
    assert cm.message_count() == 0
    cm.add_exchange("q1", "a1")
    assert cm.message_count() == 2
    cm.add_exchange("q2", "a2")
    assert cm.message_count() == 4


class TestGetContext:
  def test_empty_returns_empty(self, mem_file):
    cm = ConversationMemory(mem_file, recent_count=3)
    assert cm.get_context() == ""

  def test_returns_recent_messages(self, mem_file):
    cm = ConversationMemory(mem_file, recent_count=2)
    cm.add_exchange("q1", "a1")
    cm.add_exchange("q2", "a2")
    ctx = cm.get_context()
    assert "q1" in ctx
    assert "a2" in ctx
    assert "Recent messages" in ctx

  def test_truncates_long_messages(self, mem_file):
    cm = ConversationMemory(mem_file, recent_count=3, max_context_message_chars=20)
    cm.add_exchange("x" * 100, "y" * 100)
    ctx = cm.get_context()
    assert "x" * 20 in ctx
    assert "x" * 100 not in ctx
    assert "..." in ctx

  def test_includes_summary_when_present(self, mem_file):
    cm = ConversationMemory(mem_file, recent_count=3)
    data = {"summary": "User asked about BTC trades", "messages": []}
    mem_file.write_text(json.dumps(data))
    ctx = cm.get_context()
    assert "Previous conversation summary" in ctx
    assert "User asked about BTC trades" in ctx

  def test_only_shows_recent_window(self, mem_file):
    cm = ConversationMemory(mem_file, recent_count=1)
    cm.add_exchange("old_question", "old_answer")
    cm.add_exchange("new_question", "new_answer")
    ctx = cm.get_context()
    assert "new_question" in ctx
    assert "new_answer" in ctx
    assert "old_question" not in ctx


class TestCompaction:
  def test_needs_compaction_false_when_within_limit(self, mem_file):
    cm = ConversationMemory(mem_file, recent_count=3)
    cm.add_exchange("q1", "a1")
    cm.add_exchange("q2", "a2")
    cm.add_exchange("q3", "a3")
    assert not cm.needs_compaction()

  def test_needs_compaction_true_when_over_limit(self, mem_file):
    cm = ConversationMemory(mem_file, recent_count=2)
    cm.add_exchange("q1", "a1")
    cm.add_exchange("q2", "a2")
    cm.add_exchange("q3", "a3")
    assert cm.needs_compaction()

  def test_compact_moves_old_to_summary(self, mem_file):
    cm = ConversationMemory(mem_file, recent_count=2)
    cm.add_exchange("q1", "a1")
    cm.add_exchange("q2", "a2")
    cm.add_exchange("q3", "a3")

    def mock_summarizer(existing_summary, old_messages):
      return f"Summary of {len(old_messages)} messages"

    cm.compact(mock_summarizer)
    data = json.loads(mem_file.read_text())
    assert data["summary"] == "Summary of 2 messages"
    assert len(data["messages"]) == 4  # recent_count=2 => 4 messages kept
    assert data["messages"][0]["content"] == "q2"

  def test_compact_incorporates_existing_summary(self, mem_file):
    cm = ConversationMemory(mem_file, recent_count=1)
    data = {
      "summary": "Old summary",
      "messages": [
        {"role": "user", "content": "q1", "ts": "t1"},
        {"role": "assistant", "content": "a1", "ts": "t1"},
        {"role": "user", "content": "q2", "ts": "t2"},
        {"role": "assistant", "content": "a2", "ts": "t2"},
      ],
    }
    mem_file.write_text(json.dumps(data))

    received = {}

    def mock_summarizer(existing_summary, old_messages):
      received["summary"] = existing_summary
      received["messages"] = old_messages
      return "New combined summary"

    cm.compact(mock_summarizer)
    assert received["summary"] == "Old summary"
    assert len(received["messages"]) == 2
    data = json.loads(mem_file.read_text())
    assert data["summary"] == "New combined summary"
    assert len(data["messages"]) == 2

  def test_compact_preserves_messages_on_summarizer_failure(self, mem_file):
    cm = ConversationMemory(mem_file, recent_count=1)
    cm.add_exchange("q1", "a1")
    cm.add_exchange("q2", "a2")

    def failing_summarizer(existing_summary, old_messages):
      raise RuntimeError("LLM down")

    cm.compact(failing_summarizer)
    data = json.loads(mem_file.read_text())
    assert len(data["messages"]) == 4
    assert data["summary"] == ""

  def test_compact_noop_when_within_limit(self, mem_file):
    cm = ConversationMemory(mem_file, recent_count=3)
    cm.add_exchange("q1", "a1")

    called = {"count": 0}

    def mock_summarizer(existing_summary, old_messages):
      called["count"] += 1
      return "summary"

    cm.compact(mock_summarizer)
    assert called["count"] == 0


class TestClear:
  def test_clear_resets_everything(self, mem_file):
    cm = ConversationMemory(mem_file, recent_count=3)
    cm.add_exchange("q1", "a1")
    data = {"summary": "old stuff", "messages": json.loads(mem_file.read_text())["messages"]}
    mem_file.write_text(json.dumps(data))

    cm.clear()
    data = json.loads(mem_file.read_text())
    assert data["summary"] == ""
    assert data["messages"] == []
    assert cm.message_count() == 0


class TestCorruptFile:
  def test_handles_corrupt_json(self, mem_file):
    mem_file.write_text("not json at all")
    cm = ConversationMemory(mem_file, recent_count=3)
    assert cm.get_context() == ""
    assert cm.message_count() == 0
    cm.add_message("user", "hello")
    assert cm.message_count() == 1

  def test_handles_missing_keys(self, mem_file):
    mem_file.write_text("{}")
    cm = ConversationMemory(mem_file, recent_count=3)
    assert cm.message_count() == 0
    ctx = cm.get_context()
    assert ctx == ""
