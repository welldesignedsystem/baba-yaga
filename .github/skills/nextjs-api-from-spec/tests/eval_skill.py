#!/usr/bin/env python3
"""
Eval suite for the nextjs-api-from-spec skill.

Given an OpenAPI 3 spec, invokes the skill (Claude Code), then runs
all eval layers against the generated output.

Layers:
  L1 — 15 deterministic code checks (structural + invariants)
  L2 — Model-graded semantic judgment (3 criteria)
  L4 — Golden dataset runner with baseline/gate
  L5 — Statistical sampling (N runs)
  L6 — Human review export

Usage:
  uv run python .github/skills/nextjs-api-from-spec/tests/eval_skill.py \\
    --spec pet-store.yaml [--runs 3] [--baseline | --gate]
"""

import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import yaml
from pathlib import Path

HERE = Path(__file__).resolve().parent
SKILL_DIR = HERE.parent
SPECS_DIR = HERE

sys.path.insert(0, str(HERE.parent.parent.parent))
sys.path.insert(0, str(HERE.parent.parent.parent / "src"))

from llm import openrouter_chat_model

BASELINE_FILE = HERE / "eval-baseline.json"
REVIEW_FILE = HERE / "human-review.csv"

# ── Skill prompt ────────────────────────────────────────────

SKILL_SYSTEM_PROMPT = """You are an expert Next.js developer. Given an OpenAPI 3 spec,
generate a full Next.js App Router API scaffold:

1. TypeScript types in types/index.ts
2. Zod validation schemas in schemas/index.ts  
3. Next.js App Router route handlers in app/api/...
4. Typed fetch client in lib/api-client.ts
5. Barrel exports in each directory's index.ts

Rules:
- Use ESM import/export
- No `any` — prefer `unknown` + Zod inference
- Never use eval, require, or hardcode secrets
- Always validate external input with Zod (.parse or .safeParse)
- Every route handler must have JSDoc
- Client reads NEXT_PUBLIC_API_URL env var or defaults to /api
- Client throws ApiError on non-2xx responses
- Output the file tree structure and file contents"""


# ── Invoke the skill ─────────────────────────────────────────


def invoke_skill(spec_path: Path, temperature: float = 0.0) -> str:
    """Call the LLM skill and return the raw generated output."""
    spec_text = spec_path.read_text()
    prompt = f"Generate a full Next.js API scaffold from this OpenAPI spec:\n\n```yaml\n{spec_text}\n```"
    model = openrouter_chat_model(temperature=temperature)
    response = model.invoke([
        {"role": "system", "content": SKILL_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ])
    return response.content.strip()


def parse_generated_files(raw_output: str) -> dict[str, str]:
    """Parse skill output into {filepath: content} dict.
    
    Expects markdown code blocks with file paths as headers,
    e.g.:
      **`types/index.ts`**
      ```typescript
      ...
      ```
    """
    files = {}
    current_path = None
    current_lines = []
    in_code_block = False

    for line in raw_output.split("\n"):
        # Detect file path header: **`path/to/file`**
        m = re.match(r'\*\*`([^`]+)`\*\*', line.strip())
        if m:
            if current_path and current_lines:
                files[current_path] = "\n".join(current_lines)
            current_path = m.group(1)
            current_lines = []
            in_code_block = False
            continue

        # Detect ```lang or ```
        if line.strip().startswith("```"):
            if in_code_block:
                in_code_block = False
            else:
                in_code_block = True
            continue

        if in_code_block and current_path:
            current_lines.append(line.rstrip())

    if current_path and current_lines:
        files[current_path] = "\n".join(current_lines)

    return files


# ── Layer 1: Deterministic checks (15 checks) ────────────────


def check_tsc_no_emit(files: dict[str, str]) -> float:
    """Check 1: generated code compiles with tsc --noEmit."""
    # Write files to a temp dir and run tsc
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        for path, content in files.items():
            dest = tmpdir / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content)

        # Create minimal tsconfig.json
        tsconfig = {
            "compilerOptions": {
                "target": "ES2022",
                "module": "ESNext",
                "moduleResolution": "bundler",
                "strict": True,
                "noEmit": True,
                "jsx": "preserve",
                "skipLibCheck": True,
                "baseUrl": ".",
                "paths": {"@/*": ["./*"]},
            },
            "include": ["**/*.ts"],
        }
        (tmpdir / "tsconfig.json").write_text(json.dumps(tsconfig))
        # Add package.json with deps so tsc can resolve
        (tmpdir / "package.json").write_text(
            json.dumps({
                "name": "eval-skill",
                "private": True,
                "dependencies": {
                    "next": "^14.0.0",
                    "zod": "^3.22.0",
                },
            })
        )
        # Add minimal next-env.d.ts
        (tmpdir / "next-env.d.ts").write_text(
            '/// <reference types="next" />\n/// <reference types="next/types/global" />\n'
        )

        result = subprocess.run(
            ["npx", "tsc", "--noEmit", "--project", str(tmpdir / "tsconfig.json")],
            capture_output=True, text=True, timeout=30,
            cwd=tmpdir,
        )
        return 1.0 if result.returncode == 0 else 0.0


def check_esm_imports(files: dict[str, str]) -> float:
    """Check 2: all imports use ESM syntax (no require)."""
    for path, content in files.items():
        if "require(" in content:
            return 0.0
    return 1.0


def check_route_file_per_path(files: dict[str, str], spec: dict) -> float:
    """Check 3: each OpenAPI path has a corresponding route.ts."""
    paths = spec.get("paths", {})
    for openapi_path in paths:
        # Convert {petId} to [petId]
        nextjs_path = re.sub(r"\{(\w+)\}", r"[\1]", openapi_path)
        expected_file = f"app/api{nextjs_path}/route.ts"
        if expected_file not in files:
            return 0.0
    return 1.0


def check_named_export_matches_method(files: dict[str, str], spec: dict) -> float:
    """Check 4: export function name matches the HTTP method."""
    paths = spec.get("paths", {})
    for openapi_path, methods in paths.items():
        nextjs_path = re.sub(r"\{(\w+)\}", r"[\1]", openapi_path)
        route_file = f"app/api{nextjs_path}/route.ts"
        content = files.get(route_file, "")
        if not content:
            continue
        for method in methods:
            if method == "parameters":
                continue
            upper = method.upper()
            expected = f"export async function {upper}"
            if expected not in content:
                # Check for export function (not async)
                if f"export function {upper}" not in content:
                    return 0.0
    return 1.0


def check_jsdoc_on_handlers(files: dict[str, str]) -> float:
    """Check 5: every exported function has JSDoc."""
    for path, content in files.items():
        if not path.startswith("app/api/") or not path.endswith("/route.ts"):
            continue
        # Find all exported functions
        exports = re.findall(r"export (async )?function (\w+)", content)
        for _, func_name in exports:
            # Find the function's start and check for preceding JSDoc
            idx = content.find(f"export async function {func_name}")
            if idx == -1:
                idx = content.find(f"export function {func_name}")
            if idx == -1:
                return 0.0
            prefix = content[max(0, idx - 200):idx].strip()
            if "/**" not in prefix or "*/" not in prefix:
                return 0.0
    return 1.0


def check_zod_schema_per_type(files: dict[str, str], spec: dict) -> float:
    """Check 6: for each named schema, a Zod schema exists."""
    schemas = spec.get("components", {}).get("schemas", {})
    if not schemas:
        return 1.0  # no schemas to check
    schemas_content = ""
    for path, content in files.items():
        if "schemas/index.ts" in path or path.endswith("schemas/index.ts"):
            schemas_content = content
            break
    if not schemas_content and schemas:
        # Check any file in schemas/
        for path, content in files.items():
            if "schemas" in path and path.endswith(".ts"):
                schemas_content += content + "\n"
    if not schemas_content:
        return 0.0

    for schema_name in schemas:
        pattern = re.escape(schema_name) + r"Schema"
        if not re.search(pattern, schemas_content):
            return 0.0
    return 1.0


def check_no_any(files: dict[str, str]) -> float:
    """Check 7: no `any` type usage."""
    for path, content in files.items():
        if not path.endswith(".ts"):
            continue
        for line in content.split("\n"):
            stripped = line.strip()
            # Skip comments
            if stripped.startswith("//") or stripped.startswith("/*"):
                continue
            # Check for " any " token (not "any" in identifiers)
            if re.search(r'\bany\b', stripped) and "unknown" not in stripped:
                # Be careful with false positives like "many", "company"
                if re.search(r'(?<![a-zA-Z])any(?![a-zA-Z])', stripped):
                    return 0.0
    return 1.0


def check_no_eval_require_secrets(files: dict[str, str]) -> float:
    """Check 8: no eval, require, or hardcoded secrets."""
    for path, content in files.items():
        if not path.endswith(".ts"):
            continue
        if "eval(" in content or " require(" in content:
            return 0.0
        # Check for hardcoded secrets
        if re.search(r'["\']sk-[A-Za-z0-9]{20,}["\']', content):
            return 0.0
        if re.search(r'["\']AKIA[0-9A-Z]{16}["\']', content):
            return 0.0
    return 1.0


def check_all_input_validated(files: dict[str, str]) -> float:
    """Check 9: all route handlers use Zod .parse() or .safeParse()."""
    for path, content in files.items():
        if not path.startswith("app/api/") or not path.endswith("/route.ts"):
            continue
        # If the handler has a requestBody or query params, it should validate
        # We check that .parse( or .safeParse( appears at least once
        if ".parse(" not in content and ".safeParse(" not in content:
            # Allow if the handler is a simple GET with no params
            # but be strict: prefer validation
            if "request." in content:
                return 0.0
    return 1.0


def check_barrel_exports(files: dict[str, str]) -> float:
    """Check 10: barrel exports (types/index.ts, schemas/index.ts)."""
    expected = ["types/index.ts", "schemas/index.ts"]
    for exp in expected:
        if exp not in files:
            return 0.0
    return 1.0


def check_client_function_per_operation(files: dict[str, str], spec: dict) -> float:
    """Check 11: a client function exists for each operationId."""
    client_content = ""
    for path, content in files.items():
        if path.endswith("api-client.ts") or path.endswith("client.ts"):
            client_content = content
            break
    if not client_content:
        return 0.0

    paths = spec.get("paths", {})
    for path, methods in paths.items():
        for method, details in methods.items():
            if method == "parameters":
                continue
            op_id = details.get("operationId", "")
            if not op_id:
                op_id = f"{method}_{path.replace('/', '_')}"
            if f"export async function {op_id}" not in client_content:
                if f"async function {op_id}" not in client_content:
                    return 0.0
    return 1.0


def check_client_throws_api_error(files: dict[str, str]) -> float:
    """Check 12: client throws ApiError on non-2xx."""
    for path, content in files.items():
        if path.endswith("api-client.ts") or path.endswith("client.ts"):
            if "throw new ApiError" in content or "throw new Error" in content:
                if "!res.ok" in content or "!response.ok" in content:
                    return 1.0
    return 0.0


def check_base_url_from_env(files: dict[str, str]) -> float:
    """Check 13: client reads baseUrl from env var."""
    for path, content in files.items():
        if path.endswith("api-client.ts") or path.endswith("client.ts"):
            if "NEXT_PUBLIC_API_URL" in content:
                return 1.0
    return 0.0


def check_no_process_env_in_handlers(files: dict[str, str]) -> float:
    """Check 14 (invariant): no handler calls process.env directly."""
    for path, content in files.items():
        if path.startswith("app/api/") and path.endswith("/route.ts"):
            if "process.env" in content:
                return 0.0
    return 1.0


def check_wraps_in_next_response(files: dict[str, str]) -> float:
    """Check 15 (invariant): all responses use NextResponse.json()."""
    for path, content in files.items():
        if path.startswith("app/api/") and path.endswith("/route.ts"):
            # Should use NextResponse.json, not bare new Response or res.send
            if "new Response(" in content or ".send(" in content:
                return 0.0
    return 1.0


ALL_L1_CHECKS = [
    ("tsc_no_emit", check_tsc_no_emit),
    ("esm_imports", check_esm_imports),
    ("route_file_per_path", check_route_file_per_path),
    ("named_export_matches_method", check_named_export_matches_method),
    ("jsdoc_on_handlers", check_jsdoc_on_handlers),
    ("zod_schema_per_type", check_zod_schema_per_type),
    ("no_any", check_no_any),
    ("no_eval_require_secrets", check_no_eval_require_secrets),
    ("all_input_validated", check_all_input_validated),
    ("barrel_exports", check_barrel_exports),
    ("client_function_per_operation", check_client_function_per_operation),
    ("client_throws_api_error", check_client_throws_api_error),
    ("base_url_from_env", check_base_url_from_env),
    ("no_process_env_in_handlers", check_no_process_env_in_handlers),
    ("wraps_in_next_response", check_wraps_in_next_response),
]


def run_l1(files: dict[str, str], spec: dict) -> dict:
    """Run all 15 deterministic checks, return {check_name: score}."""
    scores = {}
    for name, fn in ALL_L1_CHECKS:
        try:
            if name in ("route_file_per_path", "named_export_matches_method",
                        "zod_schema_per_type", "client_function_per_operation"):
                scores[name] = fn(files, spec)
            elif name == "tsc_no_emit":
                scores[name] = fn(files)
            else:
                scores[name] = fn(files)
        except Exception as e:
            print(f"  L1 check '{name}' error: {e}", file=sys.stderr)
            scores[name] = 0.0
    return scores


# ── Layer 2: Model-graded ──────────────────────────────────


L2_JUDGE_PROMPT = """You are evaluating code generated by an LLM skill for a Next.js API scaffold.
Score 0.0–1.0 on three criteria — output ONLY valid JSON:

{{"schema_correctness": 0.0, "handler_logic": 0.0, "client_ergonomics": 0.0}}

- schema_correctness: Do the Zod schemas accurately reflect the OpenAPI spec's schema types and constraints? (e.g. string vs number, required vs optional, enums, nullable)
- handler_logic: Does each route handler correctly use its path params, query params, and request body? Does it return the appropriate status code?
- client_ergonomics: Is the fetch client pleasant to use — typed params, sensible defaults, URL encoding handled?

OpenAPI spec title: {spec_title}
Generated files: {file_listing}"""


def run_l2(files: dict[str, str], spec: dict) -> dict:
    """Run model-graded judgment on the generated output."""
    file_listing = json.dumps(list(files.keys()), indent=2)
    spec_title = spec.get("info", {}).get("title", "untitled")
    prompt = L2_JUDGE_PROMPT.format(spec_title=spec_title, file_listing=file_listing)

    model = openrouter_chat_model(temperature=0.0)
    response = model.invoke(prompt)
    text = response.content.strip()
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        scores = json.loads(text)
    except json.JSONDecodeError:
        return {"schema_correctness": 0.0, "handler_logic": 0.0, "client_ergonomics": 0.0}

    # Ensure all keys present
    for k in ("schema_correctness", "handler_logic", "client_ergonomics"):
        scores.setdefault(k, 0.0)
    return scores


# ── Full run ─────────────────────────────────────────────────


def load_spec(spec_path: Path) -> dict | None:
    """Safely load YAML spec, return None for malformed inputs."""
    try:
        return yaml.safe_load(spec_path.read_text())
    except yaml.YAMLError:
        return None


def run_single(spec_path: Path, temperature: float = 0.0) -> dict:
    """Run all layers on a single spec, return scored result."""
    spec = load_spec(spec_path)
    if spec is None or not isinstance(spec, dict):
        # Malformed YAML — skill should reject gracefully
        print(f"  Malformed spec {spec_path.name}, checking graceful rejection...")
        raw = invoke_skill(spec_path, temperature=temperature)
        # Expected: skill should return an error message, not code
        files = parse_generated_files(raw)
        # Score based on whether it detected the error
        has_no_code = len(files) == 0
        has_error_msg = any(w in raw.lower() for w in ["error", "invalid", "malformed", "couldn't", "unable"])
        l1_scores = {
            "skill_rejected_malformed_input": 1.0 if (has_no_code or has_error_msg) else 0.0,
        }
        l1_overall = statistics.mean(list(l1_scores.values()))
        l2_scores = {}
        l2_overall = 0.0
        return {
            "spec": spec_path.name,
            "l1": l1_scores,
            "l1_overall": round(l1_overall, 3),
            "l2": l2_scores,
            "l2_overall": round(l2_overall, 3),
            "overall": round(l1_overall, 3),
            "files": files,
            "raw_output": raw,
        }

    print(f"  Invoking skill on {spec_path.name}...")
    raw = invoke_skill(spec_path, temperature=temperature)
    files = parse_generated_files(raw)
    print(f"  Generated {len(files)} files")

    l1_scores = run_l1(files, spec)
    l1_overall = statistics.mean(l1_scores.values()) if l1_scores else 0.0

    l2_scores = run_l2(files, spec)
    l2_overall = statistics.mean(l2_scores.values()) if l2_scores else 0.0

    return {
        "spec": spec_path.name,
        "l1": l1_scores,
        "l1_overall": round(l1_overall, 3),
        "l2": l2_scores,
        "l2_overall": round(l2_overall, 3),
        "overall": round((l1_overall + l2_overall) / 2, 3),
        "files": files,
        "raw_output": raw,
    }


# ── Baseline / Gate (L4) ────────────────────────────────────


def load_baseline() -> dict:
    if BASELINE_FILE.exists():
        return json.loads(BASELINE_FILE.read_text())
    return {}


def save_baseline(results: list[dict]):
    baseline = {r["spec"]: r["overall"] for r in results}
    BASELINE_FILE.write_text(json.dumps(baseline, indent=2) + "\n")
    print(f"  Saved baseline to {BASELINE_FILE}")


def gate_results(results: list[dict], baseline: dict):
    failed = False
    for r in results:
        mean = r["overall"]
        prev = baseline.get(r["spec"], 0.0)
        delta = mean - prev
        status = "PASS" if delta >= 0 else "REGRESS"
        if delta < 0:
            failed = True
        print(f"    {status:7s}  {r['spec']:30s}  {mean:.3f}  (was {prev:.3f}, Δ {delta:+.3f})")
    if failed:
        print("\n  ❌ REGRESSION DETECTED — gate failed")
        sys.exit(1)
    print("\n  ✅ All scores at or above baseline — gate passed")


# ── Human review export (L6) ────────────────────────────────


def export_human_review(results: list[dict], sample_rate: float = 0.5):
    import csv, random
    sampled = random.sample(results, max(1, int(len(results) * sample_rate)))
    with open(REVIEW_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["spec", "l1_overall", "l2_overall", "overall", "human_score", "human_note"])
        for r in sampled:
            writer.writerow([r["spec"], r["l1_overall"], r["l2_overall"], r["overall"], "", ""])
    print(f"  Exported {len(sampled)}/{len(results)} cases to {REVIEW_FILE}")


# ── CLI ─────────────────────────────────────────────────────


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Eval suite for nextjs-api-from-spec skill")
    parser.add_argument("--spec", type=str, help="Path to OpenAPI spec file")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs (L5 sampling)")
    parser.add_argument("--baseline", action="store_true", help="Record L4 baseline")
    parser.add_argument("--gate", action="store_true", help="Gate against L4 baseline")
    parser.add_argument("--export-human", action="store_true", help="Export L6 human review CSV")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM temperature")
    args = parser.parse_args()

    # Determine which specs to run
    if args.spec:
        spec_paths = [Path(args.spec)]
    else:
        # Run all yaml specs in the tests directory
        spec_paths = sorted(SPECS_DIR.glob("*.yaml"))

    if not spec_paths:
        print("No spec files found", file=sys.stderr)
        sys.exit(1)

    print(f"Running eval on {len(spec_paths)} spec(s) × {args.runs} run(s) each")

    # L5: Statistical sampling — run each spec N times
    all_results = []
    for spec_path in spec_paths:
        print(f"\n{'='*60}")
        print(f"Spec: {spec_path.name}")
        print(f"{'='*60}")
        run_results = []
        for i in range(args.runs):
            if args.runs > 1:
                print(f"\n  Run {i+1}/{args.runs}:")
            result = run_single(spec_path, temperature=args.temperature)
            run_results.append(result["overall"])

        overalls = [r["overall"] for r in run_results] if len(run_results) > 0 else [0.0]
        mean = statistics.mean(overalls)
        stdev = statistics.stdev(overalls) if len(overalls) > 1 else 0.0
        pass_rate = sum(1 for s in overalls if s >= 0.8) / len(overalls) if overalls else 0.0

        summary = {
            "spec": spec_path.name,
            "overall": round(mean, 3),
            "stdev": round(stdev, 3),
            "pass_rate": round(pass_rate, 3),
            "n": len(overalls),
        }
        all_results.append(summary)

        # Print per-spec summary
        if args.runs > 1:
            print(f"\n  Summary for {spec_path.name}:")
            print(f"    mean={mean:.3f}  stdev={stdev:.3f}  pass_rate={pass_rate:.3f}")

    # Summary table
    print(f"\n{'='*60}")
    print(f"{'Spec':25s} {'Overall':>8s} {'Stdev':>6s} {'Pass-rate':>10s}  {'N':>4s}")
    print("-" * 60)
    for r in all_results:
        print(f"{r['spec']:25s} {r['overall']:>8.3f} {r['stdev']:>6.3f} {r['pass_rate']:>10.3f}  {r['n']:>4d}")

    # L4: Baseline / gate
    if args.baseline:
        save_baseline(all_results)
    elif args.gate:
        baseline = load_baseline()
        if not baseline:
            print("No baseline found. Run with --baseline first.")
            sys.exit(1)
        gate_results(all_results, baseline)

    # L6: Export for human review
    if args.export_human:
        export_human_review(all_results)


if __name__ == "__main__":
    main()
