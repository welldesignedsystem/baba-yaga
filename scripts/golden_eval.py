"""Golden dataset regression eval.

Usage:
    uv run python scripts/golden_eval.py              # run, show scores
    uv run python scripts/golden_eval.py --baseline   # record baseline
    uv run python scripts/golden_eval.py --gate       # gate against baseline
"""

import argparse
import os
import sys

import json

from eval import GOLDEN_DATASET, generate_synthetic_cases
from eval import gate_results, load_baseline, save_baseline, run_suite


def main():
    parser = argparse.ArgumentParser(description="Golden dataset regression eval")
    parser.add_argument("--baseline", action="store_true", help="Record baseline scores")
    parser.add_argument("--gate", action="store_true", help="Gate against existing baseline")
    parser.add_argument("--samples", type=int, default=3, help="Runs per case (default: 3)")
    parser.add_argument("--generate", action="store_true", help="Generate synthetic cases from source, then run them")
    args = parser.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("OPENROUTER_API_KEY not set")
        sys.exit(1)

    if args.generate:
        print("Generating synthetic cases from source...")
        cases = generate_synthetic_cases()
        print(f"Generated {len(cases)} cases:\n{json.dumps(cases, indent=2)}\n")

    print(f"Running {len(GOLDEN_DATASET)} golden cases × {args.samples} samples each...\n")
    results = run_suite(n=args.samples)

    for r in results:
        print(f"  {r['id']:30s}  mean={r['mean']:.3f}  stdev={r['stdev']:.3f}")

    if args.baseline:
        save_baseline(results)
    elif args.gate:
        baseline = load_baseline()
        if not baseline:
            print("No baseline found. Run with --baseline first.")
            sys.exit(1)
        gate_results(results, baseline)
    else:
        print("\n(Pass --gate to check against baseline, or --baseline to record)")


if __name__ == "__main__":
    main()
