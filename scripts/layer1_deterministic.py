#!/usr/bin/env python3
"""
Layer 1 — Deterministic / Structural Checks.

Each scorer is a plain Python function with zero dependencies (stdlib only).
No LLM calls, no API calls, no judge model.
"""

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

HERE = Path(__file__).parent.parent


# ── Scorers (pure Python, no deps) ──────────────────────────


def score_contains(output, must_contain):
    return 1.0 if all(w in output.lower() for w in must_contain) else 0.0


def score_excludes(output, must_not_contain):
    return 0.0 if any(w in output.lower() for w in must_not_contain) else 1.0


def score_max_words(output, max_words):
    return 1.0 if len(output.split()) <= max_words else 0.0


def score_valid_json(output):
    try:
        json.loads(output)
        return 1.0
    except json.JSONDecodeError:
        return 0.0


SCORERS = {
    "must_contain": lambda o, c: score_contains(o, c["must_contain"]),
    "must_not_contain": lambda o, c: score_excludes(o, c["must_not_contain"]),
    "max_words": lambda o, c: score_max_words(o, c["max_words"]),
    "expects_valid_json": lambda o, c: score_valid_json(o),
}


# ── Golden dataset ──────────────────────────────────────────


GOLDEN_DATASET = [
    {
        "id": "capital-france",
        "prompt": "What is the capital of France? Answer in one word.",
        "must_contain": ["paris"],
        "max_words": 5,
    },
    {
        "id": "capital-japan",
        "prompt": "What is the capital of Japan? Answer in one word.",
        "must_contain": ["tokyo"],
        "max_words": 5,
    },
    {
        "id": "meaning-of-life",
        "prompt": "What is the meaning of life? Answer in 10 words or fewer.",
        "max_words": 15,
    },
    {
        "id": "python-list-comprehension",
        "prompt": "Write a Python list comprehension that squares all even numbers from 0 to 20.",
        "must_contain": ["**2", "range", "if", "%"],
        "must_not_contain": ["import"],
    },
    {
        "id": "json-output",
        "prompt": 'Return only valid JSON: {"name": "Alice", "age": 30} but with age incremented by 1. Output ONLY the JSON.',
        "expects_valid_json": True,
    },
]


DEMO_OUTPUTS = {
    "capital-france": "Paris",
    "capital-japan": "Tokyo",
    "meaning-of-life": "42",
    "python-list-comprehension": (
        "[x**2 for x in range(21) if x % 2 == 0]"
    ),
    "json-output": '{"name": "Alice", "age": 31}',
}


# ── Runner ──────────────────────────────────────────────────


def score_case(case, output):
    scores = {}
    for key, scorer in SCORERS.items():
        if key in case or (key == "expects_valid_json" and case.get(key)):
            scores[key] = scorer(output, case)
    overall = statistics.mean(scores.values()) if scores else 0.0
    return {"id": case["id"], "overall": round(overall, 3), "scores": scores}


def run_suite(outputs):
    results = []
    for case in GOLDEN_DATASET:
        output = outputs[case["id"]]
        result = score_case(case, output)
        result["output_snippet"] = output[:120]
        results.append(result)
    return results


# ── Report ──────────────────────────────────────────────────


def print_report(results):
    print(f"{'ID':30s} {'Overall':>8s}  {'Output snippet'}")
    print("-" * 80)
    for r in results:
       print(f"{r['id']:30s} {r['overall']:>8.3f}  {r['output_snippet']}")
    print()
    overalls = [r["overall"] for r in results]
    print(f"Mean across all cases: {statistics.mean(overalls):.3f}")


# ── CLI ─────────────────────────────────────────────────────


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Layer 1 — Deterministic / Structural Checks")
    parser.add_argument("--live", action="store_true", help="Call a real model instead of using demo outputs")
    args = parser.parse_args()

    if args.live:
        try:
            from src.llm import openrouter_chat_model
        except ImportError:
            print("error: --live requires src/llm.py (install deps and set OPENROUTER_API_KEY)")
            sys.exit(1)
        model = openrouter_chat_model(temperature=0.0)
        outputs = {}
        for case in GOLDEN_DATASET:
            response = model.invoke(case["prompt"])
            outputs[case["id"]] = response.content.strip()
    else:
        outputs = DEMO_OUTPUTS

    results = run_suite(outputs)
    print_report(results)


if __name__ == "__main__":
    main()
