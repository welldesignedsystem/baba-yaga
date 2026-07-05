"""Analyze Claude Code session transcripts for productivity metrics.

Session logs are JSONL files (one JSON object per line, a conversation turn)
stored at ~/.claude/projects/<url-encoded-project-path>/<session-id>.jsonl.

Usage:
    uv run python scripts/analyze_sessions.py ~/.claude/projects  --report
    uv run python scripts/analyze_sessions.py ~/.claude/projects --before 2026-07-01 --after 2026-06-01
"""

import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path


def load_sessions(sessions_dir: str, before: str | None = None, after: str | None = None) -> list[list[dict]]:
    """Load all JSONL session files, optionally filtering by date range."""
    sessions = []
    for f in sorted(Path(sessions_dir).rglob("*.jsonl")):
        messages = []
        for line in f.read_text().strip().splitlines():
            if line.strip():
                messages.append(json.loads(line))
        if not messages:
            continue
        # Filter by date range based on first message timestamp or file mtime
        ts = None
        for msg in messages:
            ts_str = msg.get("timestamp") or msg.get("created_at")
            if ts_str:
                try:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    break
                except (ValueError, TypeError):
                    continue
        if ts is None:
            ts = datetime.fromtimestamp(f.stat().st_mtime)
        if before and ts >= datetime.fromisoformat(before):
            continue
        if after and ts < datetime.fromisoformat(after):
            continue
        sessions.append(messages)
    return sessions


def count_tool_calls(messages: list[dict]) -> int:
    """Count total tool_use blocks across all assistant messages."""
    count = 0
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for block in msg.get("content", []):
            if isinstance(block, dict) and block.get("type") == "tool_use":
                count += 1
    return count


def count_errors(messages: list[dict]) -> int:
    """Count tool_result blocks with is_error set to true."""
    count = 0
    for msg in messages:
        if msg.get("role") != "user":
            continue
        for block in msg.get("content", []):
            if isinstance(block, dict) and block.get("type") == "tool_result" and block.get("is_error"):
                count += 1
    return count


def total_tokens(messages: list[dict]) -> int:
    """Sum output tokens from the last assistant message's usage field."""
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and "usage" in msg:
            usage = msg["usage"]
            return usage.get("output_tokens", 0) + usage.get("input_tokens", 0)
    return 0


def report(sessions: list[list[dict]]):
    if not sessions:
        print("No sessions found.")
        return

    durations = []
    tool_call_counts = []
    error_counts = []
    token_counts = []

    for messages in sessions:
        tool_call_counts.append(count_tool_calls(messages))
        error_counts.append(count_errors(messages))
        token_counts.append(total_tokens(messages))

    total = len(sessions)
    print(f"Sessions analyzed: {total}")
    print()

    if tool_call_counts:
        avg_tools = statistics.mean(tool_call_counts)
        print(f"Avg tool calls per session:  {avg_tools:.1f}")
        print(f"  median:                   {statistics.median(tool_call_counts):.1f}")
    if token_counts:
        avg_tokens = statistics.mean(token_counts)
        print(f"Avg tokens per session:     {avg_tokens:.0f}")
    if error_counts:
        sessions_with_errors = sum(1 for e in error_counts if e > 0)
        print(f"Sessions with errors:       {sessions_with_errors} / {total} ({sessions_with_errors / total:.0%})")


def main():
    parser = argparse.ArgumentParser(description="Analyze Claude Code session transcripts")
    parser.add_argument("sessions_dir", help="Path to ~/.claude/projects/")
    parser.add_argument("--before", help="Exclude sessions after this date (YYYY-MM-DD)")
    parser.add_argument("--after", help="Exclude sessions before this date (YYYY-MM-DD)")
    parser.add_argument("--report", action="store_true", help="Print full report")
    args = parser.parse_args()

    sessions = load_sessions(args.sessions_dir, before=args.before, after=args.after)

    if args.report:
        report(sessions)
    else:
        print(f"Found {len(sessions)} sessions in {args.sessions_dir}")
        for messages in sessions:
            tool_count = count_tool_calls(messages)
            err_count = count_errors(messages)
            first = messages[0]
            summary = ""
            for block in first.get("content", []):
                if isinstance(block, dict) and block.get("type") == "text":
                    summary = block["text"][:80]
                    break
            print(f"  tools={tool_count:2d}  errors={err_count}  {summary}")


if __name__ == "__main__":
    main()
