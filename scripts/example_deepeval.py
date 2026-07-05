"""Example: using DeepEval — typed metrics, dataset management, CI-gated evals.

Run with:  uv run python scripts/example_deepeval.py
"""

import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from deepeval.metrics import BaseMetric, GEval, FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv

from src.llm import openrouter_chat_model

load_dotenv()


# ── Custom deterministic metric (same as Layer 1, as a BaseMetric) ──


class ContainsMetric(BaseMetric):
    def __init__(self, words: list[str]):
        self.words = words
        self.threshold = 0.5
        self.score = 0.0

    def measure(self, test_case: LLMTestCase):
        self.score = 1.0 if all(w in test_case.actual_output.lower() for w in self.words) else 0.0
        return self.score

    def is_successful(self):
        return (self.score or 0.0) >= self.threshold


# ── Dataset ─────────────────────────────────────────────────


def build_dataset() -> list[LLMTestCase]:
    model = openrouter_chat_model(temperature=0.0)
    cases = [
        {"input": "What is the capital of France?", "expected": "Paris"},
        {"input": "What is 2 + 2?", "expected": "4"},
        {"input": "What color is the sky on a clear day?", "expected": "blue"},
    ]
    dataset = []
    for c in cases:
        response = model.invoke(c["input"])
        dataset.append(LLMTestCase(
            input=c["input"],
            actual_output=response.content.strip(),
            expected_output=c["expected"],
        ))
    return dataset


# ── Run ─────────────────────────────────────────────────────


def main():
    print("Building dataset (calling model)...")
    dataset = build_dataset()

    # Deterministic metric
    contains_metric = ContainsMetric(words=["paris", "4", "blue"])

    # Model-graded metrics (LLM-as-judge, calls model internally)
    geval = GEval(
        name="Correctness",
        criteria="Determine if the actual output is factually correct given the input and expected output.",
        model=openrouter_chat_model(temperature=0.0),
    )
    faithfulness = FaithfulnessMetric(
        model=openrouter_chat_model(temperature=0.0),
    )

    print(f"\n{'Case':30s} {'Contains':>10s} {'GEval':>10s} {'Faithful':>10s} {'Mean':>8s}")
    print("-" * 70)
    all_scores = []
    for tc in dataset:
        contains_metric.measure(tc)
        geval.measure(tc)
        faithfulness.measure(tc)
        scores = {
            "contains": contains_metric.score,
            "geval": geval.score,
            "faithfulness": faithfulness.score,
        }
        overall = statistics.mean(scores.values())
        all_scores.append(overall)
        print(f"{tc.input[:28]:30s} {scores['contains']:>10.3f} {scores['geval']:>10.3f} "
              f"{scores['faithfulness']:>10.3f} {overall:>8.3f}")

    print(f"\nDataset mean: {statistics.mean(all_scores):.3f}")


if __name__ == "__main__":
    main()
