#!/usr/bin/env python3
"""
Layer 5 — Statistical Sampling.

A single run at temperature > 0 tells you almost nothing.
Run each case N times and look at the DISTRIBUTION of scores,
not just one output.
"""

import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))


# ── Scorers (same as Layer 1, inlined for independence) ─────


def score_contains(output, words):
    return 1.0 if all(w in output.lower() for w in words) else 0.0

def score_excludes(output, words):
    return 0.0 if any(w in output.lower() for w in words) else 1.0

def score_max_words(output, limit):
    return 1.0 if len(output.split()) <= limit else 0.0

def score_valid_json(output):
    try:
        json.loads(output)
        return 1.0
    except json.JSONDecodeError:
        return 0.0


# ── Demo — simulate N runs with variance ────────────────────


def simulate_run(case_id: str) -> float:
    """Simulate a single run returning an overall score.

    In practice you would call your real model here.
    Three cases with different variance profiles to illustrate sampling:
    - stable: always 1.0 (deterministic output)
    - medium: ~85% pass rate (some variance)
    - noisy: ~65% pass rate (high variance)
    """
    import random
    random.seed()
    profiles = {
        "stable":  lambda: 1.0,
        "medium":  lambda: 1.0 if random.random() < 0.85 else 0.0,
        "noisy":   lambda: 1.0 if random.random() < 0.65 else 0.0,
    }
    fn = profiles.get(case_id, profiles["medium"])
    return fn()


# ── Statistics ──────────────────────────────────────────────


def summarize(scores: list[float]) -> dict:
    n = len(scores)
    if n == 0:
        return {"n": 0, "mean": 0.0, "stdev": 0.0, "pass_rate": 0.0}
    mean = statistics.mean(scores)
    stdev = statistics.stdev(scores) if n > 1 else 0.0
    pass_rate = sum(1 for s in scores if s >= 0.8) / n
    return {
        "n": n,
        "mean": round(mean, 3),
        "stdev": round(stdev, 3),
        "pass_rate": round(pass_rate, 3),
    }


# ── Report ──────────────────────────────────────────────────


def print_report(results: list[dict]):
    print(f"{'Case':20s} {'N':>4s} {'Mean':>6s} {'Stdev':>6s} {'Pass-rate':>10s}  {'Interpretation'}")
    print("-" * 75)
    for r in results:
        s = r["stats"]
        interp = interpret(s)
        print(f"{r['id']:20s} {s['n']:>4d} {s['mean']:>6.3f} {s['stdev']:>6.3f} {s['pass_rate']:>10.3f}  {interp}")
    print()
    # Compounding effect for multi-step tasks
    print("Multi-step compounding (per-step reliability → trajectory reliability):")
    for steps in [3, 5, 10]:
        per_step = results[0]["stats"]["mean"]
        trajectory = per_step ** steps
        print(f"  {steps} steps @ {per_step:.0%} per-step → {trajectory:.0%} trajectory reliability")


def interpret(s: dict) -> str:
    if s["mean"] >= 0.98:
        return "stable"
    if s["stdev"] <= 0.1:
        return "consistent"
    if s["mean"] >= 0.8 and s["stdev"] > 0.2:
        return "high variance — investigate"
    if s["mean"] < 0.7:
        return "unreliable — needs improvement"
    return "moderate"


# ── CLI ─────────────────────────────────────────────────────


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Layer 5 — Statistical Sampling")
    parser.add_argument("--runs", type=int, default=10, help="Number of runs per case (default: 10)")
    args = parser.parse_args()

    cases = ["stable", "medium", "noisy"]
    results = []
    for cid in cases:
        scores = [simulate_run(cid) for _ in range(args.runs)]
        results.append({"id": cid, "stats": summarize(scores)})

    print_report(results)


if __name__ == "__main__":
    main()
