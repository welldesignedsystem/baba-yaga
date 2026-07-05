"""Example: using hypothesis for property-based testing of LLM outputs.

Generates random inputs and checks invariants that must hold for ALL
outputs, not just specific cases.

Run with:  uv run python scripts/example_hypothesis.py
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

try:
    from hypothesis import given, strategies as st, settings, Phase
except ImportError:
    print("error: hypothesis not installed — run `uv add hypothesis`")
    sys.exit(1)

from src.llm import openrouter_chat_model


# ── Invariants ──────────────────────────────────────────────


def invariant_no_secrets(output: str) -> bool:
    """Never leak API keys, tokens, or credentials."""
    import re
    patterns = [
        r"sk-[A-Za-z0-9]{20,}",
        r"AKIA[0-9A-Z]{16}",
        r"-----BEGIN (RSA |EC )?PRIVATE KEY-----",
    ]
    return not any(re.search(p, output) for p in patterns)


def invariant_within_word_limit(output: str, limit: int = 100) -> bool:
    """Never exceed the word limit."""
    return len(output.split()) <= limit


def invariant_valid_json_when_requested(output: str) -> bool:
    """If the output looks like JSON, it should be valid JSON."""
    import json
    stripped = output.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            json.loads(stripped)
            return True
        except json.JSONDecodeError:
            return False
    return True


# ── Hypothesis test ─────────────────────────────────────────


@given(
    topic=st.text(min_size=1, max_size=100).filter(lambda t: t.strip() != ""),
    style=st.sampled_from(["concise", "detailed", "json"]),
)
@settings(phases=[Phase.generate], max_examples=5)
def test_output_invariants(topic: str, style: str):
    """Property: for any valid input, the output must satisfy all invariants."""
    model = openrouter_chat_model(temperature=0.5)
    style_prompts = {
        "concise": "Answer in 3 words or fewer: ",
        "detailed": "Explain in detail: ",
        "json": 'Return valid JSON with key "answer": ',
    }
    prompt = style_prompts[style] + topic
    response = model.invoke(prompt)
    output = response.content.strip()

    assert invariant_no_secrets(output), f"Secret leaked for topic={topic!r}"
    assert invariant_within_word_limit(output), f"Too long for topic={topic!r}"
    assert invariant_valid_json_when_requested(output), f"Invalid JSON for topic={topic!r}"


if __name__ == "__main__":
    test_output_invariants()
