#!/usr/bin/env python3
"""
Full-stack eval for an OpenAPI-to-Python-client-generator skill.

Tests every layer of the eval pyramid against a single concrete skill:
given an OpenAPI 3 spec, generate a typed Python httpx client with
dataclass models, retry logic, error types, and docstrings.

Usage:
    uv run python scripts/skills/test_openapi_client_generator.py              # demo mode
    uv run python scripts/skills/test_openapi_client_generator.py --live       # call real model
    uv run python scripts/skills/test_openapi_client_generator.py --samples 10 # statistical sampling
    uv run python scripts/skills/test_openapi_client_generator.py --gate       # gate against baseline
    uv run python scripts/skills/test_openapi_client_generator.py --baseline   # record baseline
    uv run python scripts/skills/test_openapi_client_generator.py --export-csv # human review export
    uv run python scripts/skills/test_openapi_client_generator.py --live --judge  # + LLM-as-judge
"""

import ast
import csv
import json
import os
import re
import statistics
import sys
import textwrap
from pathlib import Path

# Allow importing from baba-yaga/src (file is at scripts/skills/*.py, so go up 3 levels)
HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))


# ═══════════════════════════════════════════════════════════════════════════════
# 1. FIXTURES — OpenAPI specs the skill will read
# ═══════════════════════════════════════════════════════════════════════════════
# These are the "skills" inputs — real OpenAPI 3.0 specs that
# the model reads and generates a client from. Each exercises
# different complexity: flat CRUD, nested schemas with auth,
# and a trivial edge case.

PET_STORE_SPEC = """
openapi: "3.0.0"
info:
  title: Pet Store
  version: "1.0"
paths:
  /pets:
    get:
      operationId: listPets
      summary: List all pets
      responses:
        "200":
          content:
            application/json:
              schema:
                type: array
                items:
                  $ref: "#/components/schemas/Pet"
    post:
      operationId: createPet
      summary: Create a pet
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: "#/components/schemas/NewPet"
      responses:
        "201":
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/Pet"
  /pets/{petId}:
    get:
      operationId: getPetById
      summary: Get a pet by ID
      parameters:
        - name: petId
          in: path
          required: true
          schema:
            type: integer
      responses:
        "200":
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/Pet"
components:
  schemas:
    Pet:
      type: object
      properties:
        id:
          type: integer
        name:
          type: string
        tag:
          type: string
    NewPet:
      type: object
      properties:
        name:
          type: string
        tag:
          type: string
"""

ECOMMERCE_SPEC = """
openapi: "3.0.0"
info:
  title: E-Commerce API
  version: "2.0"
servers:
  - url: https://api.example.com/v2
paths:
  /products:
    get:
      operationId: listProducts
      summary: List products with optional category filter
      parameters:
        - name: category
          in: query
          schema:
            type: string
      responses:
        "200":
          content:
            application/json:
              schema:
                type: array
                items:
                  $ref: "#/components/schemas/Product"
  /products/{productId}:
    get:
      operationId: getProduct
      summary: Get a single product
      parameters:
        - name: productId
          in: path
          required: true
          schema:
            type: string
      responses:
        "200":
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/Product"
    patch:
      operationId: updateProduct
      summary: Update product fields
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: "#/components/schemas/ProductUpdate"
      responses:
        "200":
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/Product"
  /orders:
    post:
      operationId: createOrder
      summary: Place a new order
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: "#/components/schemas/NewOrder"
      responses:
        "201":
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/Order"
components:
  schemas:
    Product:
      type: object
      properties:
        id:
          type: string
        name:
          type: string
        price:
          type: number
        category:
          type: string
    ProductUpdate:
      type: object
      properties:
        name:
          type: string
        price:
          type: number
        category:
          type: string
    NewOrder:
      type: object
      properties:
        productId:
          type: string
        quantity:
          type: integer
        shippingAddress:
          type: string
    Order:
      type: object
      properties:
        id:
          type: string
        productId:
          type: string
        quantity:
          type: integer
        total:
          type: number
        status:
          type: string
"""

MINIMAL_SPEC = """
openapi: "3.0.0"
info:
  title: Ping API
  version: "0.1"
paths:
  /ping:
    get:
      operationId: ping
      summary: Health check
      responses:
        "200":
          description: OK
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Demo outputs — expected generated clients used when --live is NOT passed
# ═══════════════════════════════════════════════════════════════════════════════
# These simulate what a good model would generate. In demo mode the
# deterministic checks (L1) run against these to show the eval pipeline
# working end-to-end without needing an API key.

PET_STORE_OUTPUT = """
import httpx
from dataclasses import dataclass
from typing import Optional


@dataclass
class Pet:
    id: int
    name: str
    tag: Optional[str] = None


@dataclass
class NewPet:
    name: str
    tag: Optional[str] = None


class PetStoreClient:
    \"\"\"Client for the Pet Store API.\"\"\"

    def __init__(self, base_url: str, timeout: float = 30.0) -> None:
        \"\"\"Initialise the client with a base URL and timeout.\"\"\"
        self._client = httpx.Client(base_url=base_url, timeout=timeout)

    def list_pets(self) -> list[Pet]:
        \"\"\"List all pets.\"\"\"
        resp = self._client.get("/pets")
        resp.raise_for_status()
        return [Pet(**item) for item in resp.json()]

    def create_pet(self, new_pet: NewPet) -> Pet:
        \"\"\"Create a pet.\"\"\"
        resp = self._client.post(
            "/pets",
            json={"name": new_pet.name, "tag": new_pet.tag},
        )
        resp.raise_for_status()
        return Pet(**resp.json())

    def get_pet_by_id(self, pet_id: int) -> Pet:
        \"\"\"Get a pet by ID.\"\"\"
        resp = self._client.get(f"/pets/{pet_id}")
        resp.raise_for_status()
        return Pet(**resp.json())
"""

ECOMMERCE_OUTPUT = """
import httpx
from dataclasses import dataclass
from typing import Optional


@dataclass
class Product:
    id: str
    name: str
    price: float
    category: str


@dataclass
class ProductUpdate:
    name: Optional[str] = None
    price: Optional[float] = None
    category: Optional[str] = None


@dataclass
class NewOrder:
    product_id: str
    quantity: int
    shipping_address: str


@dataclass
class Order:
    id: str
    product_id: str
    quantity: int
    total: float
    status: str


class ECommerceClient:
    \"\"\"Client for the E-Commerce API.\"\"\"

    def __init__(self, base_url: str = "https://api.example.com/v2", timeout: float = 30.0) -> None:
        \"\"\"Initialise the client with a base URL and timeout.\"\"\"
        self._client = httpx.Client(base_url=base_url, timeout=timeout)

    def list_products(self, category: Optional[str] = None) -> list[Product]:
        \"\"\"List products with optional category filter.\"\"\"
        params = {}
        if category is not None:
            params["category"] = category
        resp = self._client.get("/products", params=params)
        resp.raise_for_status()
        return [Product(**item) for item in resp.json()]

    def get_product(self, product_id: str) -> Product:
        \"\"\"Get a single product.\"\"\"
        resp = self._client.get(f"/products/{product_id}")
        resp.raise_for_status()
        return Product(**resp.json())

    def update_product(self, product_id: str, update: ProductUpdate) -> Product:
        \"\"\"Update product fields.\"\"\"
        resp = self._client.patch(
            f"/products/{product_id}",
            json={k: v for k, v in {
                "name": update.name,
                "price": update.price,
                "category": update.category,
            }.items() if v is not None},
        )
        resp.raise_for_status()
        return Product(**resp.json())

    def create_order(self, order: NewOrder) -> Order:
        \"\"\"Place a new order.\"\"\"
        resp = self._client.post(
            "/orders",
            json={
                "productId": order.product_id,
                "quantity": order.quantity,
                "shippingAddress": order.shipping_address,
            },
        )
        resp.raise_for_status()
        return Order(**resp.json())
"""

DEMO_OUTPUTS = {
    "pet-store": PET_STORE_OUTPUT.strip(),
    "ecommerce": ECOMMERCE_OUTPUT.strip(),
    "minimal": (
        "import httpx\n\n\n"
        "class PingClient:\n"
        '    """Client for the Ping API."""\n\n'
        "    def __init__(self, base_url: str, timeout: float = 30.0) -> None:\n"
        '        """Initialise the client with a base URL and timeout."""\n'
        "        self._client = httpx.Client(base_url=base_url, timeout=timeout)\n\n"
        "    def ping(self) -> dict:\n"
        '        """Health check."""\n'
        "        resp = self._client.get(\"/ping\")\n"
        "        resp.raise_for_status()\n"
        "        return resp.json()\n"
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 1 — Deterministic / Structural Checks
# ═══════════════════════════════════════════════════════════════════════════════
# These are the "cheapest" checks — plain Python, no LLM needed.
# They verify structural properties of the generated code.
# Everything here is a pure function: same output → same score every time.


def check_valid_python(output: str) -> float:
    """1.0 if the output compiles as valid Python, 0.0 otherwise."""
    try:
        compile(output, "<string>", "exec")
        return 1.0
    except SyntaxError:
        return 0.0


def check_classes_exist(output: str, expected_classes: list[str]) -> float:
    """1.0 if all expected class names appear in the output."""
    return 1.0 if all(c in output for c in expected_classes) else 0.0


def check_methods_exist(output: str, expected_operations: int) -> float:
    """1.0 if the output has at least the expected number of method definitions."""
    count = len(re.findall(r"\bdef \w+", output))
    return 1.0 if count >= expected_operations else 0.0


def check_imports(output: str, required_imports: list[str]) -> float:
    """1.0 if all required imports are present (e.g. httpx, dataclasses)."""
    return 1.0 if all(imp in output for imp in required_imports) else 0.0


def check_no_banned_patterns(output: str, banned: list[str]) -> float:
    """1.0 if no banned patterns appear (eval, exec, os.system, etc.)."""
    return 0.0 if any(p in output for p in banned) else 1.0


def check_all_methods_have_docstrings(output: str) -> float:
    """1.0 if every function/method has a docstring as its first statement.

    Uses the ast module to parse and walk the tree rather than fragile regex.
    """
    try:
        tree = ast.parse(output)
    except SyntaxError:
        return 0.0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.body:
                return 0.0
            first = node.body[0]
            if not isinstance(first, ast.Expr):
                return 0.0
            if not isinstance(first.value, ast.Constant):
                return 0.0
            if not isinstance(first.value.value, str):
                return 0.0
    return 1.0


def check_type_annotations(output: str) -> float:
    """1.0 if every function parameter and return has a type annotation.

    Uses ast to check that every def has -> return annotation and every
    parameter (except self/cls) has : type annotation.
    __init__ is excluded from the return annotation check since it
    implicitly returns None and Python does not require the annotation.
    """
    try:
        tree = ast.parse(output)
    except SyntaxError:
        return 0.0
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Skip return annotation check for __init__ (implicit None)
            if node.name != "__init__" and node.returns is None:
                return 0.0
            for arg in node.args.args:
                if arg.arg not in ("self", "cls") and arg.annotation is None:
                    return 0.0
            for arg in node.args.kwonlyargs:
                if arg.annotation is None:
                    return 0.0
    return 1.0


def score_layer1(output: str, case: dict) -> dict:
    """Run all Layer 1 checks and return per-check scores."""
    return {
        "valid_python": check_valid_python(output),
        "classes_exist": check_classes_exist(output, case["expected_classes"]),
        "methods_exist": check_methods_exist(output, case["expected_operations"]),
        "imports": check_imports(output, case.get("required_imports", ["httpx"])),
        "no_banned_patterns": check_no_banned_patterns(output, ["eval(", "exec(", "os.system"]),
        "docstrings": check_all_methods_have_docstrings(output),
        "type_annotations": check_type_annotations(output),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 2 — Model-Graded Evaluation (LLM-as-judge)
# ═══════════════════════════════════════════════════════════════════════════════
# These check semantic properties that cannot be verified with code alone.
# Only runs with --live --judge since it requires a second LLM call.

JUDGE_PROMPT = """You are evaluating a generated API client. Score it 0.0–1.0.

Output ONLY valid JSON — no other text:
{{"docstring_quality": 0.0, "error_handling": 0.0, "idiomatic_python": 0.0}}

- docstring_quality: do the docstrings accurately describe what each method does?
- error_handling: does the client handle errors appropriately (raise_for_status, timeouts)?
- idiomatic_python: is the code natural for modern Python with httpx and dataclasses?

OpenAPI spec title: {spec_title}

User prompt: Generate a Python client from this OpenAPI spec.

Generated client:
{output}"""


def score_layer2(output: str, spec_title: str) -> dict:
    """Call an LLM judge to score the generated client on semantic quality.

    One LLM call per evaluation. This is why Layer 2 is "expensive"
    compared to Layer 1 — every eval run costs a model call on top of
    the generation call.
    """
    from src.llm import openrouter_chat_model

    model = openrouter_chat_model(temperature=0.0)
    resp = model.invoke(JUDGE_PROMPT.format(spec_title=spec_title, output=output[:3000]))
    text = resp.content.strip()
    text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"docstring_quality": 0.0, "error_handling": 0.0, "idiomatic_python": 0.0}


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 3 — Property-Based / Invariant Checks
# ═══════════════════════════════════════════════════════════════════════════════
# These check properties that must hold across ALL generated clients
# regardless of which OpenAPI spec was used. Unlike L1 which checks
# "does this specific feature exist?", L3 checks "does any output
# violate a safety or correctness invariant?"


def check_no_hardcoded_base_url(output: str) -> float:
    """1.0 if base URLs are configurable, not hardcoded in method bodies.

    A hardcoded 'https://api.example.com' inside a method body is a
    red flag — the URL should come from a constructor parameter.
    Default values in __init__ are fine.
    """
    lines = output.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if "https://" in stripped or "http://" in stripped:
            # Accept URLs as default parameter values in __init__
            if "def __init__" in stripped:
                continue
            # Accept URLs inside httpx.Client(base_url=...) in __init__
            if "base_url" in stripped and "httpx.Client" in stripped:
                continue
            return 0.0
    return 1.0


def check_no_bare_http_calls(output: str) -> float:
    """1.0 if all httpx calls use timeout (not bare get/post/put/patch/delete).

    The generated client should always configure timeout, either at
    the Client level or on individual calls. This is a safety invariant —
    bare HTTP calls can hang indefinitely.
    """
    if "httpx.Client" in output and "timeout" not in output:
        return 0.0
    return 1.0


def check_no_dangerous_calls(output: str) -> float:
    """1.0 if the output contains no eval/exec/subprocess/__import__ calls."""
    dangerous = ["eval(", "exec(", "subprocess.", "os.system(", "__import__("]
    return 0.0 if any(d in output for d in dangerous) else 1.0


INVARIANTS = [
    ("no_hardcoded_base_url", check_no_hardcoded_base_url),
    ("no_bare_http_calls", check_no_bare_http_calls),
    ("no_dangerous_calls", check_no_dangerous_calls),
]


def score_layer3(output: str) -> dict:
    """Run all Layer 3 invariant checks — always deterministic, always runs."""
    return {name: fn(output) for name, fn in INVARIANTS}


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 4 — Golden Dataset
# ═══════════════════════════════════════════════════════════════════════════════
# A curated set of representative OpenAPI specs, each with expected
# structural properties and scoring criteria. This is what you'd
# extend over time as you add more spec types.

GOLDEN_DATASET = [
    {
        "id": "pet-store",
        "spec": PET_STORE_SPEC,
        "spec_title": "Pet Store",
        "expected_classes": ["Pet", "NewPet", "PetStoreClient"],
        "expected_operations": 3,
        "required_imports": ["httpx", "dataclasses"],
    },
    {
        "id": "ecommerce",
        "spec": ECOMMERCE_SPEC,
        "spec_title": "E-Commerce API",
        "expected_classes": ["Product", "Order", "ECommerceClient"],
        "expected_operations": 4,
        "required_imports": ["httpx", "dataclasses"],
    },
    {
        "id": "minimal",
        "spec": MINIMAL_SPEC,
        "spec_title": "Ping API",
        "expected_classes": ["PingClient"],
        "expected_operations": 1,
        "required_imports": ["httpx"],
    },
]

BASELINE_FILE = HERE / "eval-baseline-openapi-client.json"


def evaluate_case(case: dict, output: str, run_layer2: bool = False) -> dict:
    """Score a single generated client across all applicable layers.

    The overall score is the mean of all individual checks across
    both L1 and L3. L2 (judge) is optional since it costs a model call.
    """
    result = {"id": case["id"], "spec_title": case["spec_title"]}

    # Layer 1 — always run (deterministic, free, fast)
    result["layer1"] = score_layer1(output, case)

    # Layer 3 — always run (invariants are also deterministic)
    result["layer3"] = score_layer3(output)

    # Layer 2 — optional (requires model call)
    if run_layer2:
        result["layer2"] = score_layer2(output, case["spec_title"])

    # Composite: mean of every check across all layers
    all_scores = list(result["layer1"].values()) + list(result["layer3"].values())
    if run_layer2:
        all_scores += list(result.get("layer2", {}).values())
    result["overall"] = round(statistics.mean(all_scores), 3) if all_scores else 0.0
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Model call — delegates to LLM when --live, otherwise returns demo outputs
# ═══════════════════════════════════════════════════════════════════════════════


def generate_client(spec_yaml: str, spec_id: str, live: bool = False) -> str:
    """Generate a Python client from an OpenAPI spec.

    In demo mode (default): returns pre-written expected output.
    In live mode: calls the model with the spec and returns raw response.
    """
    if not live:
        return DEMO_OUTPUTS.get(spec_id, DEMO_OUTPUTS["minimal"])

    from src.llm import openrouter_chat_model

    prompt = (
        "You are generating a Python API client from the following "
        "OpenAPI 3.0 spec. Requirements:\n"
        "- Use httpx.Client for HTTP calls\n"
        "- Use dataclasses for request/response models\n"
        "- Define error classes per status code family\n"
        "- Add docstrings to every method\n"
        "- Add type annotations to all parameters and return types\n"
        "- Read base_url from constructor parameter\n"
        "- Include timeout configuration\n"
        "- Do NOT use eval, exec, or os.system\n\n"
        f"OpenAPI spec:\n{spec_yaml}\n\n"
        "Output ONLY the Python code, no explanation."
    )
    model = openrouter_chat_model(temperature=0.0)
    return model.invoke(prompt).content.strip()


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 5 — Statistical Sampling
# ═══════════════════════════════════════════════════════════════════════════════


def run_suite(n: int = 1, live: bool = False, run_layer2: bool = False) -> list[dict]:
    """Run the full golden dataset N times and return aggregated results.

    When n > 1, this activates Layer 5: instead of trusting a single run,
    we look at the distribution — mean, stdev, pass rate.
    """
    results = []
    for case in GOLDEN_DATASET:
        case_results = []
        for _ in range(n):
            output = generate_client(case["spec"], case["id"], live=live)
            case_results.append(evaluate_case(case, output, run_layer2=run_layer2))

        overalls = [r["overall"] for r in case_results]
        pass_rate = sum(1 for s in overalls if s >= 0.8) / len(overalls) if overalls else 0.0
        results.append({
            "id": case["id"],
            "mean": round(statistics.mean(overalls), 3),
            "stdev": round(statistics.stdev(overalls), 3) if n > 1 else 0.0,
            "pass_rate": round(pass_rate, 3),
            "n": n,
            "runs": case_results,
        })
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Baseline and gate (regression tracking for L4)
# ═══════════════════════════════════════════════════════════════════════════════


def load_baseline() -> dict:
    if BASELINE_FILE.exists():
        return json.loads(BASELINE_FILE.read_text())
    return {}


def save_baseline(results: list[dict]):
    baseline = {r["id"]: r["mean"] for r in results}
    BASELINE_FILE.write_text(json.dumps(baseline, indent=2) + "\n")
    print(f"  Saved baseline to {BASELINE_FILE}")


def gate_results(results: list[dict], baseline: dict):
    """Check every case against its recorded baseline. Fail if any regressed."""
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


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 6 — Human Review Export
# ═══════════════════════════════════════════════════════════════════════════════


def export_for_review(results: list[dict], path: str = "review-openapi-client.csv"):
    """Export eval results to CSV with an empty human_score column.

    Workflow:
    1. Run this script with --export-csv
    2. Reviewer opens the CSV, adds human scores
    3. Save as review-openapi-client-annotated.csv
    4. Compare auto_score vs human_score to find rubric gaps
    """
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "case_id", "spec_title", "run", "auto_score", "human_score", "notes",
        ])
        writer.writeheader()
        for r in results:
            for i, run in enumerate(r.get("runs", [r])):
                writer.writerow({
                    "case_id": r["id"],
                    "spec_title": run.get("spec_title", ""),
                    "run": i + 1,
                    "auto_score": run.get("overall", run.get("mean", 0.0)),
                    "human_score": "",
                    "notes": "",
                })
    print(f"  Exported {path} for human review")


# ═══════════════════════════════════════════════════════════════════════════════
# Report
# ═══════════════════════════════════════════════════════════════════════════════


def print_report(results: list[dict]):
    """Pretty-print the full pyramid breakdown for every case."""
    for r in results:
        # Get spec_title from the first run (it's stored per-run)
        spec_title = (r.get("runs") or [{}])[0].get("spec_title", "")
        print(f"\n{'='*72}")
        print(f"  Case: {r['id']}  ({spec_title})")
        print(f"{'='*72}")
        print(f"  Overall:    {r['mean']:.3f}")
        print(f"  Std dev:    {r['stdev']:.3f}")
        print(f"  Pass rate:  {r['pass_rate']:.0%}  (threshold: >= 0.80)")

        # Show per-layer breakdown from the last run
        run = r["runs"][-1] if r.get("runs") else r

        if "layer1" in run:
            l1 = run["layer1"]
            l1_mean = statistics.mean(l1.values()) if l1 else 0.0
            print(f"\n  ── Layer 1 — Deterministic: {l1_mean:.3f} ──")
            for name, score in sorted(l1.items()):
                print(f"    {name:30s} {score:.1f}")

        if "layer3" in run:
            l3 = run["layer3"]
            l3_mean = statistics.mean(l3.values()) if l3 else 0.0
            print(f"\n  ── Layer 3 — Invariants: {l3_mean:.3f} ──")
            for name, score in sorted(l3.items()):
                print(f"    {name:30s} {score:.1f}")

        if "layer2" in run:
            l2 = run["layer2"]
            l2_mean = statistics.mean(l2.values()) if l2 else 0.0
            print(f"\n  ── Layer 2 — Model-Graded: {l2_mean:.3f} ──")
            for name, score in sorted(l2.items()):
                print(f"    {name:30s} {score:.1f}")

    # Summary footer
    print(f"\n{'='*72}")
    print(f"  Summary across all cases")
    print(f"{'='*72}")
    means = [r["mean"] for r in results]
    print(f"  Mean overall:   {statistics.mean(means):.3f}")
    print(f"  Avg pass rate:  {statistics.mean([r['pass_rate'] for r in results]):.0%}")
    print(f"  Avg std dev:    {statistics.mean([r['stdev'] for r in results]):.3f}")

    # Compounding note
    avg_reliability = statistics.mean(means)
    print(f"\n  Multi-step compounding (per-step ≈ {avg_reliability:.0%}):")
    for steps in [3, 5, 10]:
        trajectory = avg_reliability ** steps
        print(f"    {steps} steps  →  {trajectory:.0%} trajectory reliability")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Full-stack eval: OpenAPI-to-Python-client generator",
    )
    parser.add_argument("--live", action="store_true",
                        help="Call a real model (requires OPENROUTER_API_KEY)")
    parser.add_argument("--samples", type=int, default=1,
                        help="Runs per case for sampling (default: 1)")
    parser.add_argument("--baseline", action="store_true",
                        help="Record baseline scores into eval-baseline-openapi-client.json")
    parser.add_argument("--gate", action="store_true",
                        help="Gate against existing baseline")
    parser.add_argument("--export-csv", action="store_true",
                        help="Export results for human review (Layer 6)")
    parser.add_argument("--judge", action="store_true",
                        help="Also run LLM-as-judge (Layer 2) — requires --live")
    args = parser.parse_args()

    if args.judge and not args.live:
        print("error: --judge requires --live")
        sys.exit(1)
    if args.baseline and args.gate:
        print("error: --baseline and --gate are mutually exclusive")
        sys.exit(1)
    if args.live and not os.environ.get("OPENROUTER_API_KEY"):
        print("error: --live requires OPENROUTER_API_KEY")
        sys.exit(1)

    print(f"\n{'='*50}")
    print(f"  OpenAPI Client Generator — Full-Stack Eval")
    print(f"{'='*50}")
    print(f"  Mode:     {'live' if args.live else 'demo'}")
    print(f"  Samples:  {args.samples}")
    print(f"  Layers:   L1 + L3" + (" + L2 (judge)" if args.judge else "") + " + L5 + L6")
    print()

    results = run_suite(n=args.samples, live=args.live, run_layer2=args.judge)

    if args.baseline:
        save_baseline(results)
    elif args.gate:
        print("  Checking against baseline...")
        baseline = load_baseline()
        if not baseline:
            print("  No baseline found. Run with --baseline first.")
            sys.exit(1)
        gate_results(results, baseline)

    if args.export_csv and results:
        export_for_review(results)

    print_report(results)


if __name__ == "__main__":
    main()
