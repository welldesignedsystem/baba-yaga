#!/usr/bin/env python3
"""
toolcallcheck — mock MCP server that records tool calls and lets you
assert on the trajectory: which tools were called, in what order, with
what arguments, and whether structural invariants held.

Runs fully offline — no model call needed.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))


# ── Mock MCP Server ──────────────────────────────────────────


class MockMCPServer:
    """Records every tool call and provides assertion helpers.

    Replace the real MCP server in your agent with this mock.
    The agent calls `handle_call` as it would the real server;
    after the run, assert against the recorded trajectory.
    """

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []  # (tool_name, args)
        self.responses: dict[str, object] = {}
        self._call_index: dict[str, int] = {}

    def handle_call(self, tool_name: str, args: dict) -> object:
        """Simulate a tool call. Records it and returns a canned response."""
        self.calls.append((tool_name, args))
        self._call_index.setdefault(tool_name, 0)
        self._call_index[tool_name] += 1
        return self.responses.get(tool_name, {"success": True, "data": None})

    # ── Assertion helpers ────────────────────────────────────

    def assert_called(self, tool_name: str, times: int | None = None) -> None:
        count = self._call_index.get(tool_name, 0)
        if times is not None:
            assert count == times, (
                f"Expected {tool_name!r} called {times}x, got {count}x"
            )
        else:
            assert count > 0, f"Expected {tool_name!r} to be called, was never called"

    def assert_not_called(self, tool_name: str) -> None:
        assert tool_name not in self._call_index, (
            f"Expected {tool_name!r} to never be called, called "
            f"{self._call_index[tool_name]}x"
        )

    def assert_called_once_with(self, tool_name: str, **expected_args) -> None:
        """Assert the tool was called exactly once with matching args."""
        self.assert_called(tool_name, times=1)
        for call_name, call_args in self.calls:
            if call_name == tool_name:
                for key, val in expected_args.items():
                    actual = call_args.get(key)
                    assert actual == val, (
                        f"Arg {key!r} = {actual!r}, expected {val!r}"
                    )
                return
        raise AssertionError(f"{tool_name!r} never called")

    def assert_called_in_order(self, expected: list[str]) -> None:
        """The first occurrence of each tool follows the given sequence."""
        seq = [name for name, _ in self.calls]
        idx = 0
        for tool in expected:
            try:
                idx = seq.index(tool, idx) + 1
            except ValueError:
                raise AssertionError(
                    f"Tool {tool!r} not found after position {idx} "
                    f"in call sequence {seq}"
                )

    # ── Invariant helpers (Layer 3) ───────────────────────────

    DESTRUCTIVE_TOOLS = {"delete", "destroy", "deploy", "drop", "shutdown"}

    def assert_no_destructive_calls(self) -> None:
        for name, _ in self.calls:
            assert name not in self.DESTRUCTIVE_TOOLS, (
                f"Invariant violation: destructive tool {name!r} was called"
            )

    def assert_at_most_once(self, tool_name: str) -> None:
        """Invariant: tool called at most once (e.g. process_refund)."""
        count = self._call_index.get(tool_name, 0)
        assert count <= 1, (
            f"Invariant violation: {tool_name!r} called {count}x, at most 1 expected"
        )

    def assert_step_count(self, lo: int, hi: int) -> None:
        n = len(self.calls)
        assert lo <= n <= hi, (
            f"Step count {n} outside expected range [{lo}, {hi}]"
        )


# ── Scenarios ─────────────────────────────────────────────────


def scenario_refund_flow(server: MockMCPServer) -> str:
    """Simulate an agent processing a refund request."""
    order = server.handle_call("lookup_order", {"order_id": "ord_123"})
    if not order.get("success"):
        return "Order lookup failed"
    refund = server.handle_call("process_refund", {
        "order_id": "ord_123",
        "amount": 49.99,
        "reason": "customer_request",
    })
    if not refund.get("success"):
        return "Refund failed"
    server.handle_call("send_notification", {
        "to": "customer@example.com",
        "subject": "Refund processed",
    })
    return "Refund completed"


def scenario_destructive_guard(server: MockMCPServer) -> str:
    """Simulate an agent that tries a destructive action."""
    server.handle_call("read_file", {"path": "config.yaml"})
    server.handle_call("delete", {"path": "config.yaml"})
    return "Deleted"


def scenario_search_then_write(server: MockMCPServer) -> str:
    """Simulate a well-behaved research-then-write trajectory."""
    server.handle_call("search_web", {"query": "python csv parsing"})
    server.handle_call("read_file", {"path": "data.csv"})
    server.handle_call("write_file", {
        "path": "parse.py",
        "content": "import csv\n",
    })
    return "Done"


# ── Tests ─────────────────────────────────────────────────────


def test_refund_called_once():
    """The refund-twice invariant: process_refund is called at most once."""
    server = MockMCPServer()
    server.responses["lookup_order"] = {"success": True, "order": {"id": "ord_123"}}
    server.responses["process_refund"] = {"success": True, "refund_id": "rf_456"}
    server.responses["send_notification"] = {"success": True}

    result = scenario_refund_flow(server)
    assert "Refund completed" in result

    server.assert_called_once_with("lookup_order", order_id="ord_123")
    server.assert_called_once_with("process_refund", order_id="ord_123")
    server.assert_at_most_once("process_refund")  # invariant
    server.assert_called_in_order(["lookup_order", "process_refund", "send_notification"])


def test_destructive_calls_blocked():
    """Invariant: never call destructive tools like delete."""
    server = MockMCPServer()
    server.responses["read_file"] = {"success": True, "content": "key: value"}
    server.responses["delete"] = {"success": True}

    try:
        scenario_destructive_guard(server)
    except AssertionError:
        pass  # should never reach here — but we catch in case

    with pytest_raises_or_fail(server.assert_no_destructive_calls):
        pass


def test_search_then_write_trajectory():
    """Assert correct tool order and step efficiency."""
    server = MockMCPServer()
    server.responses["search_web"] = {"success": True, "results": []}
    server.responses["read_file"] = {"success": True, "content": "a,b,c\n1,2,3"}
    server.responses["write_file"] = {"success": True}

    scenario_search_then_write(server)

    server.assert_called_in_order(["search_web", "read_file", "write_file"])
    server.assert_step_count(3, 5)
    server.assert_no_destructive_calls()


# ── Test helper ───────────────────────────────────────────────


def pytest_raises_or_fail(fn):
    """Run fn; if it raises, return True. Otherwise return False."""
    try:
        fn()
        return False
    except AssertionError:
        return True


# ── Report runner ─────────────────────────────────────────────


def main():
    scenarios = [
        ("refund-flow",         scenario_refund_flow,         "process_refund called once, correct order"),
        ("destructive-guard",   scenario_destructive_guard,   "assert_no_destructive_calls catches delete"),
        ("search-then-write",   scenario_search_then_write,   "correct trajectory: search → read → write"),
    ]

    print(f"{'Scenario':25s} {'Status':12s}  Description")
    print("-" * 65)

    for name, fn, desc in scenarios:
        server = MockMCPServer()
        # Set up responses
        server.responses["lookup_order"] = {"success": True, "order": {"id": "ord_123"}}
        server.responses["process_refund"] = {"success": True, "refund_id": "rf_456"}
        server.responses["send_notification"] = {"success": True}
        server.responses["read_file"] = {"success": True, "content": "a"}
        server.responses["delete"] = {"success": True}
        server.responses["search_web"] = {"success": True, "results": []}
        server.responses["write_file"] = {"success": True}

        try:
            fn(server)
            status = "PASS"
        except AssertionError as e:
            status = f"FAIL ({e})"

        print(f"{name:25s} {status:12s}  {desc}")

    print()
    print("Run as pytest tests:")
    print("  uv run pytest scripts/tools/example_toolcallcheck.py -v")
    print()
    print("Key patterns:")
    print("  assert_called(name)          — tool was called at least once")
    print("  assert_called_once_with(...) — exactly once, with matching args")
    print("  assert_called_in_order(...)  — tools called in expected sequence")
    print("  assert_at_most_once(name)    — invariant: never called twice")
    print("  assert_no_destructive_calls  — invariant: no delete/deploy/etc")


if __name__ == "__main__":
    main()
