#!/usr/bin/env python3
"""
Layer 2 — Model-Graded Evaluation (LLM-as-judge).

Uses a second LLM call to score outputs on qualitative criteria
(correctness, helpfulness, conciseness, safety).
"""

import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

JUDGE_PROMPT = """You are grading an AI assistant's response. Score it 0.0–1.0 on each criterion below.

Output ONLY valid JSON with this exact structure — no other text:
{{"correctness": 0.0, "helpfulness": 0.0, "conciseness": 0.0}}

- correctness: is the answer factually accurate?
- helpfulness: does it directly address what was asked?
- conciseness: is it free of unnecessary verbosity?

User prompt:
{prompt}

Assistant response:
{response}"""


# ── Golden cases with known outputs ─────────────────────────


EVAL_CASES = [
    {
        "id": "capital-france-good",
        "prompt": "What is the capital of France?",
        "output": "The capital of France is Paris.",
    },
    {
        "id": "capital-france-evasive",
        "prompt": "What is the capital of France?",
        "output": "That depends on what you mean by capital. France is a country in Western Europe with a rich history. There are many interesting cities in France, most notably Paris. But maybe you meant capital in terms of culture, which could be argued to be Lyon or Marseille.",
    },
    {
        "id": "capital-france-wrong",
        "prompt": "What is the capital of France?",
        "output": "London has been a major European capital for centuries.",
    },
    {
        "id": "fibonacci",
        "prompt": "Write a Python function that returns the nth Fibonacci number.",
        "output": (
            "def fib(n):\n"
            '    """Return the nth Fibonacci number."""\n'
            "    a, b = 0, 1\n"
            "    for _ in range(n):\n"
            "        a, b = b, a + b\n"
            "    return a"
        ),
    },
    {
        "id": "fibonacci-no-code",
        "prompt": "Write a Python function that returns the nth Fibonacci number.",
        "output": "The Fibonacci sequence is a series where each number is the sum of the two preceding ones, usually starting with 0 and 1.",
    },
]


# ── Judge ───────────────────────────────────────────────────


def judge_score(prompt: str, response: str) -> dict:
    from src.llm import openrouter_chat_model

    model = openrouter_chat_model(temperature=0.0)
    resp = model.invoke(JUDGE_PROMPT.format(prompt=prompt, response=response))
    text = resp.content.strip()
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"correctness": 0.0, "helpfulness": 0.0, "conciseness": 0.0}


# ── Report ──────────────────────────────────────────────────


def print_report(results):
    criteria = ["correctness", "helpfulness", "conciseness"]
    header = f"{'ID':35s}" + "".join(f"{c:>14s}" for c in criteria) + f"{'Mean':>8s}"
    print(header)
    print("-" * len(header))
    all_scores = {c: [] for c in criteria}
    for r in results:
        scores = r["scores"]
        for c in criteria:
            all_scores[c].append(scores.get(c, 0.0))
        row = f"{r['id']:35s}"
        row += "".join(f"{scores.get(c, 0.0):>14.3f}" for c in criteria)
        row += f"{r['overall']:>8.3f}"
        print(row)
    print()
    for c in criteria:
        print(f"  {c}: mean={statistics.mean(all_scores[c]):.3f}")


# ── CLI ─────────────────────────────────────────────────────


def main():
    results = []
    for case in EVAL_CASES:
        scores = judge_score(case["prompt"], case["output"])
        overall = statistics.mean(scores.values()) if scores else 0.0
        results.append({"id": case["id"], "scores": scores, "overall": round(overall, 3)})
    print_report(results)


if __name__ == "__main__":
    main()
