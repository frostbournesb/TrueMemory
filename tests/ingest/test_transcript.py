"""Tests for transcript parsing."""

import json
import tempfile
from pathlib import Path

from neuromem.ingest.transcript import parse_transcript, format_for_extraction, Message


def test_parse_json_array():
    """Test parsing a JSON array transcript (Claude Code format)."""
    transcript = json.dumps([
        {"type": "human", "content": "What's the weather like in Seattle?"},
        {"type": "assistant", "content": "I don't have real-time weather data, but Seattle typically..."},
        {"type": "human", "content": "I live in Seattle, by the way."},
        {"type": "assistant", "content": "Good to know! Seattle is a great city."},
    ])

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write(transcript)
        f.flush()
        messages = parse_transcript(f.name)

    assert len(messages) == 4
    assert messages[0].role == "human"
    assert messages[0].content == "What's the weather like in Seattle?"
    assert messages[1].role == "assistant"
    assert messages[2].content == "I live in Seattle, by the way."

    Path(f.name).unlink()


def test_parse_jsonl():
    """Test parsing a JSONL transcript."""
    lines = [
        json.dumps({"role": "user", "content": "My name is Alice"}),
        json.dumps({"role": "assistant", "content": "Nice to meet you, Alice!"}),
    ]
    transcript = "\n".join(lines)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(transcript)
        f.flush()
        messages = parse_transcript(f.name)

    assert len(messages) == 2
    # The parser normalizes "user" -> "human" so downstream code has a
    # single role name for user turns regardless of whether the transcript
    # came from Claude Code (which uses type="user") or an external tool.
    assert messages[0].role == "human"
    assert messages[0].content == "My name is Alice"

    Path(f.name).unlink()


def test_parse_plain_text():
    """Test parsing a plain text transcript."""
    transcript = """Human: I prefer dark mode for everything.
Assistant: Noted! I'll keep that in mind.
Human: Also, I use vim keybindings.
Assistant: Great, vim keybindings it is."""

    messages = parse_transcript(transcript)
    assert len(messages) == 4
    assert messages[0].role == "human"
    assert messages[0].content == "I prefer dark mode for everything."


def test_parse_content_blocks():
    """Test parsing messages with content blocks (Claude API format)."""
    transcript = json.dumps([
        {
            "type": "assistant",
            "content": [
                {"type": "text", "text": "Let me check that for you."},
                {"type": "tool_use", "name": "Read", "input": {"path": "/etc/hosts"}},
            ]
        },
    ])

    messages = parse_transcript(transcript)
    assert len(messages) == 1
    assert "Let me check that for you" in messages[0].content


def test_format_for_extraction():
    """Test formatting messages for LLM extraction."""
    messages = [
        Message(role="human", content="I live in Seattle, Washington."),
        Message(role="assistant", content="That's a great city!"),
        Message(role="tool_use", content="[tool: Read]", tool_name="Read"),
        Message(role="human", content="I prefer TypeScript over JavaScript."),
    ]

    formatted = format_for_extraction(messages)
    assert "User: I live in Seattle" in formatted
    assert "Assistant: That's a great city" in formatted
    assert "tool" not in formatted.lower() or "Read" not in formatted
    assert "User: I prefer TypeScript" in formatted


def test_empty_transcript():
    """Test handling of empty transcripts."""
    messages = parse_transcript("")
    assert len(messages) == 0

    messages = parse_transcript("[]")
    assert len(messages) == 0
