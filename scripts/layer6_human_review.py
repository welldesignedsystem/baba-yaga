#!/usr/bin/env python3
"""
Layer 6 — Human-in-the-Loop.

Samples eval outputs for manual review and merges human annotations
back into the scoring pipeline. Bridges automated evals and domain
expert review.
"""

import csv
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
REVIEW_FILE = HERE / "human-review.csv"


# ── Simulated eval results (in practice these come from the eval runner) ─


SAMPLE_RESULTS = [
    {"id": "capital-france", "prompt": "What is the capital of France?", "output": "Paris", "auto_score": 1.0},
    {"id": "capital-france-evasive", "prompt": "What is the capital of France?", "output": "That depends on what you mean by capital...", "auto_score": 0.333},
    {"id": "fibonacci", "prompt": "Write a Python function for nth Fibonacci", "output": "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a", "auto_score": 1.0},
    {"id": "fibonacci-no-code", "prompt": "Write a Python function for nth Fibonacci", "output": "The Fibonacci sequence is a series where each number...", "auto_score": 0.333},
    {"id": "json-output", "prompt": "Return valid JSON with name and age", "output": '{"name": "Alice", "age": 31}', "auto_score": 1.0},
    {"id": "json-wrong-field", "prompt": "Return valid JSON with name and age", "output": '{"name": "Alice", "age": "thirty-one"}', "auto_score": 0.667},
    {"id": "code-review", "prompt": "Review this code for bugs", "output": "There is a potential off-by-one error in the loop condition. Consider using `range(len(items))` instead of hardcoding the bound.", "auto_score": 0.833},
    {"id": "sql-injection", "prompt": "Generate a SQL query for user lookup", "output": "SELECT * FROM users WHERE id = ?", "auto_score": 1.0},
]


def export_sample(results: list[dict], sample_rate: float = 0.5):
    """Export a random sample of results for human review."""
    sampled = random.sample(results, max(1, int(len(results) * sample_rate)))
    with open(REVIEW_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "prompt", "output", "auto_score", "human_score", "human_note"])
        for r in sampled:
            writer.writerow([r["id"], r["prompt"], r["output"], r["auto_score"], "", ""])
    print(f"Exported {len(sampled)}/{len(results)} cases to {REVIEW_FILE}")


def import_annotations() -> list[dict]:
    """Import human annotations from the review CSV."""
    if not REVIEW_FILE.exists():
        print(f"Review file not found: {REVIEW_FILE}")
        return []
    annotations = []
    with open(REVIEW_FILE, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("human_score", "").strip():
                annotations.append({
                    "id": row["id"],
                    "human_score": float(row["human_score"]),
                    "human_note": row.get("human_note", ""),
                })
    return annotations


def report_disagreements(annotations: list[dict], original_results: list[dict]):
    """Compare human scores against automated scores and report disagreements."""
    results_by_id = {r["id"]: r for r in original_results}
    disagreements = []
    for ann in annotations:
        r = results_by_id.get(ann["id"])
        if not r:
            continue
        auto = r["auto_score"]
        human = ann["human_score"]
        diff = abs(auto - human)
        if diff > 0.3:
            disagreements.append({
                "id": ann["id"],
                "auto_score": auto,
                "human_score": human,
                "diff": diff,
                "note": ann["human_note"],
            })
    return disagreements


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Layer 6 — Human-in-the-Loop review workflow")
    parser.add_argument("--export", action="store_true", help="Export a sample for human review")
    parser.add_argument("--report", action="store_true", help="Show disagreement report from reviewed file")
    args = parser.parse_args()

    if args.export:
        export_sample(SAMPLE_RESULTS)
        print("Edit human-review.csv: fill in human_score (0.0–1.0) and optional human_note.")
        return

    if args.report:
        annotations = import_annotations()
        if not annotations:
            print("No annotations found. Run --export first, then fill in human-review.csv.")
            return
        disagreements = report_disagreements(annotations, SAMPLE_RESULTS)
        print(f"Annotated: {len(annotations)} cases")
        print(f"Disagreements (auto vs human diff > 0.3): {len(disagreements)}")
        print()
        if disagreements:
            for d in disagreements:
                print(f"  {d['id']:30s}  auto={d['auto_score']:.3f}  human={d['human_score']:.3f}  Δ={d['diff']:.3f}  {d['note']}")
            print()
            print("Disagreements indicate the rubric or auto-scorer needs refinement.")
        else:
            print("No significant disagreements — auto-scorer aligns with human judgment.")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
