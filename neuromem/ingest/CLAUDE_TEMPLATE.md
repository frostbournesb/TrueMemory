# Neuromem — Persistent Memory

When the `neuromem` MCP server is connected, follow these rules:

## Auto-Recall (every session)
- At the START of each conversation, call `neuromem_search` with a broad query about the user (e.g. "user preferences and context") to load relevant memories before responding.
- Before making recommendations, check memory for stored preferences.
- When the user asks anything about past conversations or "do you remember", search memory.

## Auto-Store (during conversation)
- When the user shares a personal preference, store it immediately via `neuromem_store`. Do not ask permission.
- When an important decision is made, store it.
- When the user corrects you, store the correction.
- When the user shares a fact about themselves (location, job, projects, etc.), store it.
- Write each memory as a clear, atomic statement: "Prefers bun over npm" not "The user mentioned they like bun."
- Do NOT store full conversations, large code blocks, or transient debugging context.

## Background Processing
- Memories are also extracted automatically from conversations via background processing.
- The Stop hook captures the full transcript and runs deep extraction after sessions end.
- You do NOT need to store everything manually — focus on in-conversation corrections and explicit preferences.
- The background extractor handles: personal facts, preferences, decisions, temporal facts, and technical context.

## What You Don't Need To Do
- Don't try to remember everything — the hooks capture it automatically.
- Don't store code snippets or debugging details — those live in the codebase.
- Don't store greetings or pleasantries.
- Don't duplicate-check before storing — the ingestion pipeline handles deduplication.
