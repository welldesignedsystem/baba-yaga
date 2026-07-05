"""Analyze Claude Code session logs for productivity metrics.

Usage:
    uv run python scripts/analyze_sessions.py ~/.claude/logs/  --before 2026-06-20 --after 2026-07-05
    uv run python scripts/analyze_sessions.py ~/.claude/logs/  --report
"""

import argparse
import json
import os
import statistics
from datetime import datetime, timedelta
from pathlib import Path


def load_sessions(log_dir: str, before: str | None = None, after: str | None = None) -> list[dict]:
    sessions = []
    for f in sorted(Path(log_dir).glob("*.json")):
        data = json.loads(f.read_text())
        ts = datetime.fromisoformat(data.get("started_at", "2000-01-01"))
        if before and ts >= datetime.fromisoformat(before):
            continue
        if after and ts < datetime.fromisoformat(after):
            continue
        sessions.append(data)
    return sessions


def extract_skills(sessions: list[dict]) -> list[dict]:
    skills = []
    for s in sessions:
        for e in s.get("events", []):
            if e.get("type") == "skill_invocation":
                skills.append(e)
    return skills


def report(sessions: list[dict]):
    if not sessions:
        print("No sessions found.")
        return

    skills = extract_skills(sessions)

    durations = []
    tool_counts = []
    error_counts = []
    token_counts = []
    for s in sessions:
        start = datetime.fromisoformat(s["started_at"])
        end = datetime.fromisoformat(s["ended_at"])
        durations.append((end - start).total_seconds())
        tool_counts.append(sum(1 for e in s.get("events", []) if e.get("type") == "tool_call"))
        error_counts.append(s.get("error_count", 0))
        token_counts.append(s.get("total_tokens", 0))

    total = len(sessions)
    print(f"Sessions analyzed: {total}")
    print(f"Skill invocations: {len(skills)} ({len(skills) / total:.2f}/session)" if total else "Skill invocations: 0")
    print()

    if durations:
        avg_duration = statistics.mean(durations)
        print(f"Avg task duration:  {avg_duration:.0f}s ({avg_duration / 60:.1f}m)")
        print(f"  median:           {statistics.median(durations):.0f}s")
    if tool_counts:
        avg_tools = statistics.mean(tool_counts)
        print(f"Avg tool calls:     {avg_tools:.1f}")
        print(f"  median:           {statistics.median(tool_counts):.1f}")
    if error_counts:
        error_rate = sum(1 for e in error_counts if e > 0) / total * 100
        print(f"Error rate:         {error_rate:.1f}%")
    if token_counts:
        avg_tokens = statistics.mean(token_counts)
        print(f"Avg tokens/task:    {avg_tokens:.0f}")

    if skills:
        print()
        print("Skills breakdown:")
        by_skill = {}
        for sk in skills:
            name = sk.get("skill", "unknown")
            by_skill.setdefault(name, []).append(sk)
        for name, invocations in sorted(by_skill.items(), key=lambda x: -len(x[1])):
            print(f"  {name:30s}  {len(invocations)} invocations")


def main():
    parser = argparse.ArgumentParser(description="Analyze Claude Code session logs")
    parser.add_argument("log_dir", help="Path to ~/.claude/logs/")
    parser.add_argument("--before", help="Exclude sessions after this date (YYYY-MM-DD)")
    parser.add_argument("--after", help="Exclude sessions before this date (YYYY-MM-DD)")
    parser.add_argument("--report", action="store_true", help="Print full report")
    args = parser.parse_args()

    sessions = load_sessions(args.log_dir, before=args.before, after=args.after)
    report(sessions)


if __name__ == "__main__":
    main()
