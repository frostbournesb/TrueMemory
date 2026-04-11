"""Regression tests for the real on-disk Claude Code transcript schema.

The simplified fixture at ``tests/fixtures/sample_claude_code_transcript.json``
uses a top-level ``content`` field, but real Claude Code transcripts on disk
nest the payload under ``entry["message"]["content"]`` and mix in
``file-history-snapshot``, ``progress``, ``permission-mode``, ``attachment``,
and ``last-prompt`` entries alongside actual turns. Assistant turns also
contain ``thinking`` blocks (internal chain-of-thought that must NOT leak
into fact extraction) and ``tool_use`` blocks. Tool results come back as
``type: "user"`` entries whose content is a list of ``tool_result`` blocks.

A live test once revealed the parser was silently returning 0 messages on
every real transcript because it was only looking at the top-level
``content`` key. These tests lock the fix in so that doesn't regress.
"""

from pathlib import Path

from truememory.ingest.transcript import Message, format_for_extraction, parse_transcript

FIXTURE = Path(__file__).parent / "fixtures" / "sample_real_claude_code_transcript.jsonl"


def _parse() -> list[Message]:
    """Parse the real-format fixture."""
    assert FIXTURE.exists(), f"fixture missing: {FIXTURE}"
    return parse_transcript(FIXTURE)


def test_real_format_extracts_nonzero_messages():
    """Parser must extract the conversation turns from the real schema.

    This is the core regression: before the fix, the parser only looked at
    the top-level ``content`` key and returned an empty list for real
    transcripts (all content lives under ``message.content``).
    """
    msgs = _parse()
    assert len(msgs) > 0, "parser returned zero messages on real-format transcript"


def test_real_format_message_counts_by_role():
    """The fixture should yield exactly the expected conversational turns.

    Fixture contents (conversational only):
      - 2 user turns (type: "user" -> role: "human")
      - 3 assistant turns (2 text, 1 with text + tool_use)
      - 1 tool_result turn (type: "user" with tool_result content -> role: "tool_result")

    Non-conversation entries (file-history-snapshot, progress,
    permission-mode, attachment, last-prompt) and the thinking-only
    assistant chunk should NOT appear.
    """
    msgs = _parse()
    roles = [m.role for m in msgs]

    assert roles.count("human") == 2, f"expected 2 human turns, got {roles}"
    assert roles.count("assistant") == 3, f"expected 3 assistant turns, got {roles}"
    assert roles.count("tool_result") == 1, f"expected 1 tool_result turn, got {roles}"
    assert len(msgs) == 6, f"expected 6 total turns, got {len(msgs)}: {roles}"


def test_real_format_filters_file_history_snapshot_and_progress():
    """``file-history-snapshot`` and ``progress`` entries must be dropped.

    These are Claude Code bookkeeping entries — they carry no conversation
    content and must never show up in the parsed message stream (they'd
    otherwise confuse downstream fact extraction).
    """
    msgs = _parse()
    joined = " ".join(m.content for m in msgs)

    # Content bodies from those entry types must not leak through as messages
    assert "file-history-snapshot" not in joined
    assert "trackedFileBackups" not in joined
    assert "progressId" not in joined
    assert "snapshot" not in joined

    # And no message should have a role named after those entry types
    for m in msgs:
        assert m.role not in ("file-history-snapshot", "progress")


def test_real_format_skips_thinking_blocks():
    """``thinking`` blocks are internal CoT and must not be extracted.

    The fixture includes a streaming-chunk assistant entry whose only
    content block is ``{"type": "thinking", ...}``. After the thinking
    block is skipped, that entry has no content and should drop out
    entirely. No message content should contain the thinking signature.
    """
    msgs = _parse()
    joined = " ".join(m.content for m in msgs)
    assert "SCRUBBED_SIGNATURE" not in joined, "thinking block signature leaked into messages"
    assert "thinking" not in joined.lower() or "think" in joined.lower(), (
        "'thinking' block content leaked into messages"
    )


def test_real_format_tool_use_labeled_and_kept_in_assistant_turn():
    """``tool_use`` blocks must be labeled and retained on the assistant turn.

    The fixture has one assistant turn with ``text`` + ``tool_use`` (Read).
    The parser should render the tool_use as ``[tool: Read]`` inline and
    populate ``tool_name`` on the Message.
    """
    msgs = _parse()
    tool_use_assistants = [
        m for m in msgs if m.role == "assistant" and m.tool_name
    ]
    assert len(tool_use_assistants) == 1, (
        f"expected 1 assistant turn with a tool_use, got {len(tool_use_assistants)}"
    )
    msg = tool_use_assistants[0]
    assert msg.tool_name == "Read"
    assert "[tool: Read]" in msg.content
    # The accompanying text should still be present on the same turn
    assert "Let me check" in msg.content


def test_real_format_tool_result_retagged_not_human():
    """Pure tool_result user turns must be re-tagged as role=tool_result.

    In Claude Code's schema, tool results return to the model as
    ``type: "user"`` with a list of ``tool_result`` blocks. They're model
    plumbing, not user conversation, and must NOT appear as ``role=human``
    or they'd get fed into fact extraction as if the user said them.
    """
    msgs = _parse()
    tool_results = [m for m in msgs if m.role == "tool_result"]
    assert len(tool_results) == 1
    assert "(file is empty)" in tool_results[0].content

    # And importantly: no human turn should contain the tool_result payload
    for m in msgs:
        if m.role == "human":
            assert "(file is empty)" not in m.content


def test_real_format_user_type_normalized_to_human():
    """``type: "user"`` entries (real user prompts) must become role=human.

    Downstream code treats ``human`` as the canonical user role, so the
    parser normalizes Claude Code's ``"user"`` to ``"human"``.
    """
    msgs = _parse()
    # No message should still carry the raw "user" role
    assert not any(m.role == "user" for m in msgs), (
        "raw 'user' role leaked through; should be normalized to 'human'"
    )
    # But we should definitely have human turns
    assert any(m.role == "human" for m in msgs)


def test_real_format_format_for_extraction_is_clean():
    """``format_for_extraction`` should produce clean User:/Assistant: text.

    - ``User:`` and ``Assistant:`` labels must both appear.
    - The ``tool_result`` turn must be filtered out (its content
      ``(file is empty)`` must not show up — that's model plumbing).
    - The thinking signature must not appear.
    - The ``[tool: Read]`` marker from the assistant turn's tool_use
      block is fine to include (it's part of the assistant message).
    """
    msgs = _parse()
    formatted = format_for_extraction(msgs)

    assert "User:" in formatted
    assert "Assistant:" in formatted

    # tool_result content must be filtered out entirely
    assert "(file is empty)" not in formatted
    assert "tool_use_id" not in formatted

    # thinking signature must never appear
    assert "SCRUBBED_SIGNATURE" not in formatted

    # Real user content should be present
    assert "open-source project" in formatted
    assert "SQLite" in formatted

    # No raw schema keys should leak through
    assert "parentUuid" not in formatted
    assert "file-history-snapshot" not in formatted


def test_real_format_timestamps_preserved():
    """Top-level ``timestamp`` on entries should make it onto messages."""
    msgs = _parse()
    # All conversational turns in the fixture carry an ISO timestamp
    for m in msgs:
        assert m.timestamp, f"missing timestamp on {m.role}: {m.content[:60]!r}"
        assert m.timestamp.startswith("2026-04-04T"), m.timestamp
