#!/usr/bin/env python3
"""
Layer 3 — Property-Based / Invariant Testing.

Instead of checking a specific output, check properties that must hold
across ALL outputs regardless of input variance.

Uses Hypothesis to generate edge-case inputs automatically.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

try:
    from hypothesis import given, strategies as st, settings
except ImportError:
    print("error: hypothesis is required — run `uv add hypothesis`")
    sys.exit(1)


# ── Properties under test ───────────────────────────────────


def property_no_secrets(output: str) -> bool:
    """Never leak API keys, tokens, or secret-like patterns."""
    import re
    patterns = [
        r"sk-[A-Za-z0-9]{20,}",       # OpenAI-style keys
        r"api[-_]?key['\"]?\s*[:=]\s*['\"][A-Za-z0-9_]{16,}",
        r"AKIA[0-9A-Z]{16}",           # AWS access keys
        r"-----BEGIN (RSA |EC )?PRIVATE KEY-----",
    ]
    return not any(re.search(p, output) for p in patterns)


def property_no_debug_prints(output: str) -> bool:
    """Never contain debugging print statements in generated code."""
    import re
    return not bool(re.search(r"\bprint\s*\(.*debug", output, re.IGNORECASE))


def property_tool_call_count(output: str) -> bool:
    """Tool call nestedness should not exceed a reasonable depth."""
    depth = 0
    max_depth = 0
    for ch in output:
        if ch == "{":
            depth += 1
            max_depth = max(max_depth, depth)
        elif ch == "}":
            depth -= 1
    return max_depth <= 8


INVARIANTS = [
    ("no_secrets", property_no_secrets),
    ("no_debug_prints", property_no_debug_prints),
    ("tool_call_depth", property_tool_call_count),
]


# ── Mock model output generator ─────────────────────────────


def simulate_output(topic: str, style: str) -> str:
    """Simulate an LLM output for testing invariants.

    In practice you would call your real model here.
    """
    outputs = {
        "code": (
            "def handler(event, context):\n"
            "    # TODO: implement\n"
            '    print("debug: got event", event)\n'
            "    return {'statusCode': 200}\n"
        ),
        "config": (
            '{"connection": {"host": "db.example.com", "password": "sk-secret-key-12345"}}'
        ),
        "docs": "# Installation\n\nRun `pip install mypackage` to get started.",
    }
    return outputs.get(style, outputs["docs"])


# ── Runner ──────────────────────────────────────────────────


def check_invariants(output: str) -> dict:
    results = {}
    for name, fn in INVARIANTS:
        results[name] = 1.0 if fn(output) else 0.0
    return results


# ── Hypothesis test ─────────────────────────────────────────


@given(
    topic=st.text(min_size=1, max_size=50).filter(lambda t: t.strip()),
    style=st.sampled_from(["code", "config", "docs"]),
)
@settings(max_examples=20)
def test_invariants_hold(topic: str, style: str):
    output = simulate_output(topic, style)
    for name, fn in INVARIANTS:
        assert fn(output), f"Invariant '{name}' violated for topic={topic!r} style={style!r}"


# ── CLI ─────────────────────────────────────────────────────


def main():
    import statistics

    test_cases = [
        {"id": "api-handler", "topic": "AWS Lambda handler", "style": "code"},
        {"id": "db-config", "topic": "database connection", "style": "config"},
        {"id": "docs-page", "topic": "installation guide", "style": "docs"},
    ]

    print(f"{'ID':25s} {'no_secrets':>12s} {'no_debug_prints':>17s} {'tool_call_depth':>17s}  {'Violations'}")
    print("-" * 85)
    all_results = []
    for case in test_cases:
        output = simulate_output(case["topic"], case["style"])
        scores = check_invariants(output)
        violations = [k for k, v in scores.items() if v == 0.0]
        all_results.append(scores)
        row = f"{case['id']:25s}"
        row += "".join(f"{scores.get(k, 0.0):>12.1f}" for k, _ in INVARIANTS)
        row += f"  {', '.join(violations) if violations else 'none':s}"
        print(row)

    print(f"\nRun with `hypothesis` to generate 20 random inputs:")
    print(f"  pytest scripts/layer3_property_based.py")

    for name, _ in INVARIANTS:
        scores = [r[name] for r in all_results]
        print(f"  {name}: pass-rate {statistics.mean(scores):.0%} ({len([s for s in scores if s])}/{len(scores)})")


if __name__ == "__main__":
    main()
