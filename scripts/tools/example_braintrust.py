"""Example: using Braintrust — persist eval results linked to git commits.

Run with:
  export BRAINTRUST_API_KEY=...
  uv run python scripts/example_braintrust.py

Install: uv add braintrust
"""

import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

try:
    from braintrust import Eval
except ImportError:
    print("error: braintrust not installed — run `uv add braintrust`")
    sys.exit(1)

from src.llm import openrouter_chat_model


# ── Scorers (same as Layer 1) ───────────────────────────────


def score_contains(output, words):
    return 1.0 if all(w in output.lower() for w in words) else 0.0

def score_excludes(output, words):
    return 0.0 if any(w in output.lower() for w in words) else 1.0

def score_valid_json(output):
    try:
        json.loads(output)
        return 1.0
    except json.JSONDecodeError:
        return 0.0


# ── Dataset ─────────────────────────────────────────────────


GOLDEN_DATASET = [
    {"input": "What is the capital of France? Answer in one word.", "expected": "Paris",
     "checks": {"must_contain": ["paris"]}},
    {"input": "What is the capital of Japan? Answer in one word.", "expected": "Tokyo",
     "checks": {"must_contain": ["tokyo"]}},
    {"input": 'Return valid JSON: {"name": "Alice", "age": 30}.', "expected": '{"name": "Alice", "age": 30}',
     "checks": {"expects_valid_json": True}},
]


# ── Braintrust eval task ────────────────────────────────────


def task(input_data):
    """The function under evaluation. Braintrust calls this for each case."""
    model = openrouter_chat_model(temperature=0.0)
    response = model.invoke(input_data["input"])
    return response.content.strip()


def scorer(output, expected, checks=None):
    """Custom scorer that Braintrust records."""
    scores = {}
    if not checks:
        checks = {}
    for word in checks.get("must_contain", []):
        scores["contains_" + word] = score_contains(output, [word])
    if checks.get("expects_valid_json"):
        scores["valid_json"] = score_valid_json(output)
    overall = statistics.mean(scores.values()) if scores else 1.0
    scores["overall"] = overall
    return scores


# You can also use built-in scorers:
# from braintrust import scored_by_context, scored_by_regex, scored_by_llm


def main():
    print("Running Braintrust eval...")
    results = Eval(
        "baba-yaga-eval",              # project name
        data=[{**c, "checks": c["checks"]} for c in GOLDEN_DATASET],
        task=task,
        scores=[scorer],
        metadata={"model": "openrouter"},
    )
    print(results)


if __name__ == "__main__":
    main()
