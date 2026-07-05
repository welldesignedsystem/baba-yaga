import json
import statistics
from pathlib import Path

from dotenv import load_dotenv

from llm import openrouter_chat_model

load_dotenv()

BASELINE_FILE = Path(__file__).parent.parent / "eval-baseline.json"

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


def score_contains(output: str, must_contain: list[str]) -> float:
    return 1.0 if all(w in output.lower() for w in must_contain) else 0.0


def score_excludes(output: str, must_not_contain: list[str]) -> float:
    return 0.0 if any(w in output.lower() for w in must_not_contain) else 1.0


def score_max_words(output: str, max_words: int) -> float:
    return 1.0 if len(output.split()) <= max_words else 0.0


def score_valid_json(output: str) -> float:
    try:
        json.loads(output)
        return 1.0
    except json.JSONDecodeError:
        return 0.0


def run_eval_case(case: dict) -> dict:
    model = openrouter_chat_model(temperature=0.0)
    response = model.invoke(case["prompt"])
    output = response.content.strip()

    scores = {}
    if "must_contain" in case:
        scores["must_contain"] = score_contains(output, case["must_contain"])
    if "must_not_contain" in case:
        scores["must_not_contain"] = score_excludes(output, case["must_not_contain"])
    if "max_words" in case:
        scores["max_words"] = score_max_words(output, case["max_words"])
    if case.get("expects_valid_json"):
        scores["valid_json"] = score_valid_json(output)

    overall = statistics.mean(scores.values()) if scores else 0.0
    return {
        "id": case["id"],
        "overall": round(overall, 3),
        "scores": scores,
        "output_snippet": output[:120],
    }


def run_suite(n: int = 3) -> list[dict]:
    results = []
    for case in GOLDEN_DATASET:
        case_runs = [run_eval_case(case) for _ in range(n)]
        overalls = [r["overall"] for r in case_runs]
        results.append({
            "id": case["id"],
            "mean": round(statistics.mean(overalls), 3),
            "stdev": round(statistics.stdev(overalls), 3) if n > 1 else 0.0,
        })
    return results


def generate_synthetic_cases(source_dir: str = "src") -> list[dict]:
    """Generate golden test cases from source files using the model itself.

    Reads all Python files under *source_dir*, sends them to the model with a
    prompt asking it to propose eval cases (prompt + scoring criteria), and
    returns the parsed result.  This lets you expand coverage from real code
    instead of inventing every case by hand.
    """
    import ast, os

    src_root = Path(__file__).parent.parent / source_dir
    texts = []
    for pyfile in sorted(src_root.rglob("*.py")):
        if "site-packages" in str(pyfile) or ".venv" in str(pyfile):
            continue
        texts.append(f"--- {pyfile.relative_to(src_root.parent)} ---\n{pyfile.read_text()}")

    corpus = "\n".join(texts)
    if not corpus.strip():
        return []

    prompt = (
        "You are generating a golden eval dataset for the codebase below.\n"
        "For each file, propose at most one test case that a user might prompt "
        "an LLM about that code.\n\n"
        "Output ONLY valid JSON — a list of objects with these keys:\n"
        '- "id": short kebab-case identifier\n'
        '- "prompt": what a user would ask the model about this code\n'
        '- "must_contain": list of strings the answer should contain\n'
        '- "max_words": int, max allowed word count\n\n'
        "Return an empty list if the code doesn't suggest any useful cases.\n\n"
        f"Codebase:\n{corpus[:8000]}"
    )

    model = openrouter_chat_model(temperature=0.0)
    response = model.invoke(prompt)
    output = response.content.strip()

    output = output.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        cases = json.loads(output)
    except json.JSONDecodeError:
        return []
    return cases if isinstance(cases, list) else []


def load_baseline() -> dict:
    if BASELINE_FILE.exists():
        return json.loads(BASELINE_FILE.read_text())
    return {}


def save_baseline(results: list[dict]):
    baseline = {r["id"]: r["mean"] for r in results}
    BASELINE_FILE.write_text(json.dumps(baseline, indent=2) + "\n")
    print(f"Saved baseline to {BASELINE_FILE}")


def gate_results(results: list[dict], baseline: dict):
    failed = False
    for r in results:
        mean = r["mean"]
        prev = baseline.get(r["id"], 0.0)
        delta = mean - prev
        status = "PASS" if delta >= 0 else "REGRESS"
        if delta < 0:
            failed = True
        print(f"  {status:7s}  {r['id']:30s}  {mean:.3f}  (was {prev:.3f}, Δ {delta:+.3f})")
    if failed:
        print("\n❌ REGRESSION DETECTED — gate failed")
        raise SystemExit(1)
    else:
        print("\n✅ All scores at or above baseline — gate passed")
