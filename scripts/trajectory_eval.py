#!/usr/bin/env python3
"""
Part 3 — Trajectory Evaluation.

Scoring the full sequence of tool calls, not just the final output.
Checks: tool selection correctness, argument correctness, step efficiency,
recovery behaviour, path validity.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))


# ── Mock tool call types ────────────────────────────────────


@dataclass
class ToolCall:
    name: str
    args: dict
    success: bool = True


@dataclass
class Trajectory:
    steps: list[ToolCall] = field(default_factory=list)
    final_output: str = ""


# ── Scorers ─────────────────────────────────────────────────


def score_tool_selection(traj: Trajectory, expected_tools: list[str]) -> float:
    """1.0 if every expected tool was called at least once."""
    called = {t.name for t in traj.steps}
    return 1.0 if all(t in called for t in expected_tools) else 0.0


def score_tool_order(traj: Trajectory, expected_order: list[str]) -> float:
    """1.0 if the first occurrence of each tool follows the expected order."""
    seq = [t.name for t in traj.steps]
    idx = 0
    for tool in expected_order:
        try:
            idx = seq.index(tool, idx) + 1
        except ValueError:
            return 0.0
    return 1.0


def score_step_efficiency(traj: Trajectory, expected_min: int, expected_max: int) -> float:
    """1.0 if step count is within expected range, scaling down outside."""
    n = len(traj.steps)
    if expected_min <= n <= expected_max:
        return 1.0
    if n < expected_min:
        return n / expected_min
    return max(0.0, 1.0 - (n - expected_max) * 0.1)


def score_recovery(traj: Trajectory) -> float:
    """1.0 if every failed tool call is followed by a sensible recovery."""
    for i, step in enumerate(traj.steps):
        if not step.success:
            if i + 1 >= len(traj.steps):
                return 0.0
            next_step = traj.steps[i + 1]
            if next_step.name == step.name:
                continue
            if next_step.name in ("explain", "ask_user"):
                continue
            return 0.0
    return 1.0


def score_argument_correctness(traj: Trajectory, expected_args: dict[str, dict]) -> float:
    """1.0 if all required args for each tool are present and correct."""
    matches = 0
    total = 0
    for step in traj.steps:
        if step.name not in expected_args:
            continue
        required = expected_args[step.name]
        total += 1
        if all(k in step.args and step.args[k] == v for k, v in required.items()):
            matches += 1
    return matches / total if total > 0 else 1.0


# ── Test trajectories ───────────────────────────────────────


GOOD_TRAJECTORY = Trajectory(
    steps=[
        ToolCall("read_file", {"path": "spec.md"}),
        ToolCall("search_web", {"query": "library docs"}),
        ToolCall("write_file", {"path": "output.py", "content": "pass"}),
    ],
    final_output="Implementation complete.",
)

RECOVERED_TRAJECTORY = Trajectory(
    steps=[
        ToolCall("read_file", {"path": "spec.md"}),
        ToolCall("write_file", {"path": "output.py", "content": ""}, success=False),
        ToolCall("write_file", {"path": "output.py", "content": "import os"}, success=True),
    ],
    final_output="Done after retry.",
)

BAD_TRAJECTORY = Trajectory(
    steps=[
        ToolCall("search_web", {"query": "how to do X"}),
        ToolCall("search_web", {"query": "tutorial for X"}),
        ToolCall("search_web", {"query": "example of X"}),
        ToolCall("search_web", {"query": "X library comparison"}),
        ToolCall("search_web", {"query": "X best practices"}),
        ToolCall("read_file", {"path": "spec.md"}),
    ],
    final_output="Here are some search results.",
)

TEST_CASES = [
    {
        "id": "good",
        "traj": GOOD_TRAJECTORY,
        "expected_tools": ["read_file", "search_web", "write_file"],
        "expected_order": ["read_file", "search_web", "write_file"],
        "expected_min_steps": 2,
        "expected_max_steps": 4,
        "expected_args": {"write_file": {"path": "output.py"}},
    },
    {
        "id": "recovered",
        "traj": RECOVERED_TRAJECTORY,
        "expected_tools": ["read_file", "write_file"],
        "expected_order": ["read_file", "write_file"],
        "expected_min_steps": 2,
        "expected_max_steps": 5,
        "expected_args": {"write_file": {"path": "output.py"}},
    },
    {
        "id": "bad",
        "traj": BAD_TRAJECTORY,
        "expected_tools": ["read_file", "write_file"],
        "expected_order": ["read_file"],
        "expected_min_steps": 2,
        "expected_max_steps": 4,
        "expected_args": {},
    },
]


# ── Runner ──────────────────────────────────────────────────


def evaluate_trajectory(case: dict) -> dict:
    traj = case["traj"]
    scores = {
        "tool_selection": score_tool_selection(traj, case["expected_tools"]),
        "tool_order": score_tool_order(traj, case["expected_order"]),
        "step_efficiency": score_step_efficiency(traj, case["expected_min_steps"], case["expected_max_steps"]),
        "recovery": score_recovery(traj),
        "argument_correctness": score_argument_correctness(traj, case["expected_args"]),
    }
    import statistics
    overall = statistics.mean(scores.values())
    return {"id": case["id"], "scores": scores, "overall": round(overall, 3)}


# ── Report ──────────────────────────────────────────────────


def print_report(results):
    criteria = ["tool_selection", "tool_order", "step_efficiency", "recovery", "argument_correctness"]
    header = f"{'ID':20s}" + "".join(f"{c:>22s}" for c in criteria) + f"{'Overall':>9s}"
    print(header)
    print("-" * len(header))
    for r in results:
        row = f"{r['id']:20s}"
        row += "".join(f"{r['scores'].get(c, 0.0):>22.1f}" for c in criteria)
        row += f"{r['overall']:>9.3f}"
        print(row)


# ── CLI ─────────────────────────────────────────────────────


def main():
    results = [evaluate_trajectory(c) for c in TEST_CASES]
    print_report(results)
    print()
    print("How to read:")
    print("  good      = ideal trajectory: correct tools, right order, efficient, no failures")
    print("  recovered = tool fails but agent retries successfully — recovery works")
    print("  bad       = too many search steps, missing write_file — efficiency fails")


if __name__ == "__main__":
    main()
