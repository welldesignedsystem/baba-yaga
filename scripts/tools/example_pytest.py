"""Example: using pytest for LLM eval — fixtures, parametrize, invariant checks.

Run with:  pytest scripts/example_pytest.py -v
"""

import json
import statistics
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))


# ── Scorers (deterministic, no deps) ────────────────────────


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


# ── Fixtures ────────────────────────────────────────────────


@pytest.fixture
def model():
    """Fixture that provides the model under test.

    Swap this for your real model when running against production.
    """
    from src.llm import openrouter_chat_model
    return openrouter_chat_model(temperature=0.0)


# ── Parametrized golden dataset ─────────────────────────────


GOLDEN_CASES = [
    pytest.param(
        "What is the capital of France? Answer in one word.",
        {"must_contain_scorers": (["paris"],), "max_words_scorers": (5,)},
        id="capital-france",
    ),
    pytest.param(
        "Write a Python list comprehension that squares all even numbers from 0 to 20.",
        {
            "must_contain_scorers": (["**2", "range", "if", "%"],),
            "must_not_contain_scorers": (["import"],),
        },
        id="python-list-comprehension",
    ),
    pytest.param(
        'Return only valid JSON: {"name": "Alice", "age": 30} with age incremented by 1.',
        {"valid_json_scorers": (True,)},
        id="json-output",
    ),
]


def run_checks(output: str, checks: dict) -> list[str]:
    """Run all applicable scorers, return list of failure messages."""
    failures = []
    for words in checks.get("must_contain_scorers", []):
        if not score_contains(output, words):
            failures.append(f"must_contain {words}")
    for words in checks.get("must_not_contain_scorers", []):
        if not score_excludes(output, words):
            failures.append(f"must_not_contain {words}")
    for limit in checks.get("max_words_scorers", []):
        if not score_max_words(output, limit):
            failures.append(f"max_words > {limit}")
    if checks.get("valid_json_scorers"):
        if not score_valid_json(output):
            failures.append("valid_json")
    return failures


# ── Tests ───────────────────────────────────────────────────


@pytest.mark.parametrize("prompt,checks", GOLDEN_CASES)
def test_golden_case(prompt, checks, model):
    """Run a golden case and assert all scorers pass."""
    response = model.invoke(prompt)
    output = response.content.strip()
    failures = run_checks(output, checks)
    assert not failures, f"{failures}\n  output: {output[:200]}"


# ── Invariant test with pytest fixture ──────────────────────


@pytest.fixture
def invariant_violations():
    """Post-test fixture that checks invariants across all outputs.

    Yields control to the test, then runs invariant checks on exit.
    """
    outputs = []
    yield outputs
    for output in outputs:
        assert "sk-" not in output, "Secret key leaked in output"
        assert len(output) > 0, "Empty output"


def test_invariant_no_secrets(invariant_violations):
    outputs = ["Paris", "Tokyo"]
    invariant_violations.extend(outputs)


# ── Sampling: run same case N times, report distribution ────


SAMPLING_CASES = [
    ("capital-france", "What is the capital of France? Answer in one word.",
     {"must_contain_scorers": (["paris"],)}),
]


@pytest.mark.parametrize("case_id,prompt,checks", SAMPLING_CASES)
def test_sampling_distribution(case_id, prompt, checks, model):
    """Run N times and report pass-rate instead of binary pass/fail."""
    N = 5
    outputs = [model.invoke(prompt).content.strip() for _ in range(N)]
    failure_counts = [len(run_checks(o, checks)) for o in outputs]
    pass_rate = sum(1 for f in failure_counts if f == 0) / N
    print(f"\n  {case_id}: pass-rate {pass_rate:.0%} ({sum(1 for f in failure_counts if f == 0)}/{N})")
    assert pass_rate >= 0.6, f"pass-rate too low: {pass_rate:.0%}"
