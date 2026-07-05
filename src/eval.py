import json
import re
import statistics
import subprocess
from pathlib import Path

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
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

ROUNDTRIP_DATASET = [
    {
        "id": "auth-module",
        "doc_path": "docs/design/auth.md",
        "code_path": "src/auth/",
        "check_compile": True,
        "check_signatures": ["login()", "verify_token()", "refresh()"],
        "check_exports": ["authenticate", "AuthError"],
    },
]


# ── DeepEval custom metrics ─────────────────────────────────


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


class ExcludesMetric(BaseMetric):
    def __init__(self, words: list[str]):
        self.words = words
        self.threshold = 0.5
        self.score = 0.0

    def measure(self, test_case: LLMTestCase):
        self.score = 0.0 if any(w in test_case.actual_output.lower() for w in self.words) else 1.0
        return self.score

    def is_successful(self):
        return (self.score or 0.0) >= self.threshold


class MaxWordsMetric(BaseMetric):
    def __init__(self, max_words: int):
        self.max_words = max_words
        self.threshold = 0.5
        self.score = 0.0

    def measure(self, test_case: LLMTestCase):
        self.score = 1.0 if len(test_case.actual_output.split()) <= self.max_words else 0.0
        return self.score

    def is_successful(self):
        return (self.score or 0.0) >= self.threshold


class ValidJsonMetric(BaseMetric):
    def __init__(self):
        self.threshold = 0.5
        self.score = 0.0

    def measure(self, test_case: LLMTestCase):
        try:
            json.loads(test_case.actual_output)
            self.score = 1.0
        except json.JSONDecodeError:
            self.score = 0.0
        return self.score

    def is_successful(self):
        return (self.score or 0.0) >= self.threshold


# ── Helpers ──────────────────────────────────────────────────


def build_metrics(case: dict) -> list[BaseMetric]:
    metrics = []
    if "must_contain" in case:
        metrics.append(ContainsMetric(case["must_contain"]))
    if "must_not_contain" in case:
        metrics.append(ExcludesMetric(case["must_not_contain"]))
    if "max_words" in case:
        metrics.append(MaxWordsMetric(case["max_words"]))
    if case.get("expects_valid_json"):
        metrics.append(ValidJsonMetric())
    return metrics


def score_from_case(case: dict, output: str) -> dict:
    tc = LLMTestCase(input=case["prompt"], actual_output=output)
    metrics = build_metrics(case)
    scores = {}
    for m in metrics:
        m.measure(tc)
        scores[type(m).__name__] = m.score
    overall = statistics.mean(scores.values()) if scores else 0.0
    return {"id": case["id"], "overall": round(overall, 3), "scores": scores}


# ── Runner ──────────────────────────────────────────────────


def run_eval_case(case: dict) -> dict:
    model = openrouter_chat_model(temperature=0.0)
    response = model.invoke(case["prompt"])
    result = score_from_case(case, response.content.strip())
    result["output_snippet"] = response.content.strip()[:120]
    return result


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


# ── Baseline / gate ─────────────────────────────────────────


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
    print("\n✅ All scores at or above baseline — gate passed")


# ── Synthetic case generation ───────────────────────────────


def generate_synthetic_cases(source_dir: str = "src") -> list[dict]:
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


# ── Roundtrip consistency: code ↔ docs ─────────────────────


def code_compile_score(path: str) -> float:
    try:
        result = subprocess.run(
            ["python", "-m", "py_compile", path],
            capture_output=True, text=True, timeout=10,
        )
        return 1.0 if result.returncode == 0 else 0.0
    except Exception:
        return 0.0


def extract_signatures(code: str) -> list[str]:
    sigs = []
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith("def "):
            sigs.append(stripped.removeprefix("def ").split(":")[0].strip())
    return sigs


def signature_match_score(generated_code: str, expected: list[str]) -> float:
    found = extract_signatures(generated_code)
    return 1.0 if all(s in found for s in expected) else 0.0


def doc_roundtrip_score(original_doc: str, generated_doc: str) -> float:
    original_headers = set(re.findall(r"^#{1,4}\s+", original_doc, re.MULTILINE))
    generated_headers = set(re.findall(r"^#{1,4}\s+", generated_doc, re.MULTILINE))
    if not original_headers:
        return 1.0
    intersection = original_headers & generated_headers
    return len(intersection) / len(original_headers)


def run_roundtrip_case(case: dict) -> dict:
    doc_path = Path(__file__).parent.parent / case["doc_path"]
    original_doc = doc_path.read_text() if doc_path.exists() else ""

    prompt = f"Write Python code implementing the following spec:\n\n{original_doc[:4000]}"
    model = openrouter_chat_model(temperature=0.0)
    generated_code = model.invoke(prompt).content.strip()

    compile_score = 0.0
    sig_score = 0.0
    if case.get("check_compile"):
        compile_score = code_compile_score(generated_code)
    if case.get("check_signatures"):
        sig_score = signature_match_score(generated_code, case["check_signatures"])

    prompt2 = f"Write documentation for the following Python code:\n\n{generated_code[:4000]}"
    generated_doc = model.invoke(prompt2).content.strip()

    rt_score = doc_roundtrip_score(original_doc, generated_doc)

    scores = {"code_compile": compile_score, "signature_match": sig_score, "doc_roundtrip": rt_score}
    overall = statistics.mean(scores.values()) if scores else 0.0
    return {"id": case["id"], "overall": round(overall, 3), "scores": scores}


def run_roundtrip_suite(n: int = 1) -> list[dict]:
    results = []
    for case in ROUNDTRIP_DATASET:
        case_runs = [run_roundtrip_case(case) for _ in range(n)]
        overalls = [r["overall"] for r in case_runs]
        results.append({
            "id": case["id"],
            "mean": round(statistics.mean(overalls), 3),
            "stdev": round(statistics.stdev(overalls), 3) if n > 1 else 0.0,
        })
    return results
