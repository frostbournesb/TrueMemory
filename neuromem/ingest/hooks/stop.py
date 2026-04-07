#!/usr/bin/env python3
"""
Stop Hook — Trigger Background Extraction
===========================================

Fires when a Claude Code session ends. Reads the conversation transcript
and launches the ingestion pipeline to extract and store memories.

This is the "sleep consolidation" trigger — the conversation is over,
now process it for long-term storage. The pipeline runs in a background
subprocess so it doesn't block Claude Code's shutdown.

Input (stdin JSON):
    {"session_id": "...", "transcript_path": "...", "stop_reason": "..."}

Output: None (processing happens in background)
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

log = logging.getLogger(__name__)

GATE_THRESHOLD = float(os.environ.get("NEUROMEM_GATE_THRESHOLD", "0.30"))
MIN_MESSAGES = int(os.environ.get("NEUROMEM_MIN_MESSAGES", "5"))
TRACE_DIR = Path(os.environ.get(
    "NEUROMEM_TRACE_DIR",
    str(Path.home() / ".neuromem" / "traces"),
))
LOG_DIR = Path(os.environ.get(
    "NEUROMEM_LOG_DIR",
    str(Path.home() / ".neuromem" / "logs"),
))


def _parse_args() -> argparse.Namespace:
    """Parse command-line overrides for user_id and db_path.

    Resolution order is: command-line arg > env var > empty default. The
    installer threads these flags through so multiple Claude Code profiles
    can share a single interpreter while still writing to separate DBs.
    ``parse_known_args`` is used so the hook tolerates unknown flags from
    future installer versions.
    """
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--user", default=os.environ.get("NEUROMEM_USER_ID", ""))
    p.add_argument("--db", default=os.environ.get("NEUROMEM_DB_PATH", ""))
    args, _ = p.parse_known_args()
    return args


def main():
    args = _parse_args()

    try:
        input_data = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        input_data = {}

    transcript_path = input_data.get("transcript_path", "")
    session_id = input_data.get("session_id", "unknown")

    if not transcript_path or not Path(transcript_path).exists():
        return

    # Pre-flight check: ensure our write directories exist and are writable
    # so we fail early with a clear message instead of silently losing work
    if not _writable_dirs_ok():
        return

    # Proper transcript validation: parse and count actual message objects
    # instead of substring-matching (which false-positives on content that
    # contains the literal strings "human" or "user").
    if not _has_enough_messages(transcript_path, MIN_MESSAGES):
        return

    # Run ingestion in the background so we don't block Claude Code
    _run_background_ingestion(transcript_path, session_id, args.user, args.db)


def _writable_dirs_ok() -> bool:
    """Verify the trace and log directories exist and are writable.

    Writes a short marker to ~/.neuromem/.health so that a user who is
    debugging knows the hook at least fired and checked its environment.
    Returns False (and prints to stderr) if the directories can't be used.
    """
    try:
        TRACE_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"neuromem-ingest stop hook: cannot create ~/.neuromem dirs: {e}",
              file=sys.stderr)
        return False

    # Check disk free space — abort if less than 10 MB available
    try:
        stats = shutil.disk_usage(LOG_DIR)
        if stats.free < 10 * 1024 * 1024:
            print(f"neuromem-ingest stop hook: disk full (free={stats.free} bytes)",
                  file=sys.stderr)
            return False
    except OSError:
        pass  # Non-fatal if disk_usage fails

    # Test write access with a touch file
    try:
        health_file = LOG_DIR / ".health"
        health_file.write_text("ok", encoding="utf-8")
    except OSError as e:
        print(f"neuromem-ingest stop hook: logs dir not writable: {e}",
              file=sys.stderr)
        return False

    return True


def _has_enough_messages(transcript_path: str, min_messages: int) -> bool:
    """Check whether the transcript has at least `min_messages` user turns.

    Parses the transcript properly instead of substring-counting so that
    conversations containing the literal strings "human"/"user" in content
    don't inflate the count.
    """
    try:
        content = Path(transcript_path).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return False

    if not content.strip():
        return False

    # Try JSON array first (Claude Code format)
    count = 0
    try:
        if content.lstrip().startswith("["):
            data = json.loads(content)
            if isinstance(data, list):
                for entry in data:
                    if isinstance(entry, dict):
                        role = entry.get("type") or entry.get("role") or ""
                        if role in ("human", "user"):
                            count += 1
                return count >= min_messages
    except json.JSONDecodeError:
        pass

    # Try JSONL
    try:
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if isinstance(entry, dict):
                    role = entry.get("type") or entry.get("role") or ""
                    if role in ("human", "user"):
                        count += 1
            except json.JSONDecodeError:
                continue
        if count > 0:
            return count >= min_messages
    except Exception:
        pass

    # Fall back to length heuristic for plain text
    return len(content) > min_messages * 50


def _run_background_ingestion(
    transcript_path: str,
    session_id: str,
    user_id: str,
    db_path: str,
):
    """Launch the ingestion pipeline as a background process.

    Captures stderr/stdout to a log file so silent failures are recoverable.
    Handles Windows (CREATE_NEW_PROCESS_GROUP) and POSIX (start_new_session)
    subprocess detachment.
    """
    # Log the effective config for debugging. Operators commonly wire up
    # multiple profiles and need to confirm which user/db the hook actually
    # saw after the arg-parse + env-var resolution.
    log.info(
        "stop hook: launching ingestion user=%r db=%r session=%r",
        user_id, db_path, session_id,
    )

    # Build the command
    cmd = [
        sys.executable, "-m", "neuromem.ingest.cli",
        "ingest", transcript_path,
    ]

    if user_id:
        cmd.extend(["--user", user_id])
    if db_path:
        cmd.extend(["--db", db_path])

    cmd.extend(["--threshold", str(GATE_THRESHOLD)])
    if session_id:
        cmd.extend(["--session", session_id])

    # Save trace for debugging
    trace_path = TRACE_DIR / f"{session_id}.json"
    log_path = LOG_DIR / f"{session_id}.log"
    cmd.extend(["--trace", str(trace_path)])

    # OS-specific subprocess detachment kwargs
    # POSIX: start_new_session to detach from parent's process group
    # Windows: CREATE_NEW_PROCESS_GROUP flag
    detach_kwargs: dict = {}
    if sys.platform == "win32":
        detach_kwargs["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
            | getattr(subprocess, "DETACHED_PROCESS", 0)
        )
    else:
        detach_kwargs["start_new_session"] = True

    log_file = None
    try:
        # Open the log file and hand it to the subprocess. We MUST close our
        # parent-side handle after Popen — the subprocess has its own dup'd
        # copy, so closing ours here is safe and prevents FD leaks across
        # many invocations of this hook (one per session).
        log_file = open(log_path, "a", encoding="utf-8")
        subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            close_fds=True,
            **detach_kwargs,
        )
    except Exception as e:
        # If background launch fails, try inline (blocking)
        log.warning("Background launch failed: %s, running inline", e)
        try:
            from neuromem.ingest import ingest
            result = ingest(
                transcript_path=transcript_path,
                user_id=user_id,
                db_path=db_path or None,
                gate_threshold=GATE_THRESHOLD,
                session_id=session_id,
            )
            from neuromem.ingest.pipeline import save_trace
            save_trace(result, trace_path)
        except Exception as e2:
            log.error("Inline ingestion also failed: %s", e2)
            try:
                log_path.write_text(f"Inline ingestion failed: {e2}\n", encoding="utf-8")
            except Exception:
                pass
    finally:
        # Always close our parent-side handle to prevent FD leaks.
        # The subprocess still has its own dup'd copy of the FD and will
        # continue writing to the log file normally.
        if log_file is not None:
            try:
                log_file.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
