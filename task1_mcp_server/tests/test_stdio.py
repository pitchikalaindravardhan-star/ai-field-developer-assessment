"""Drives the server as a real subprocess, speaking raw JSON-RPC over pipes.

Going through the SDK client here would hide the thing being tested: every line
on stdout has to be a valid JSON-RPC message.
"""

import json
import subprocess
import sys
import threading
from pathlib import Path

import mcp_types as types
import pytest

ROOT = Path(__file__).resolve().parents[2]
VALID_REASON = "Duplicate charge on invoice"


def run_session(messages, module="task1_mcp_server.server"):
    """Runs initialize, then the given messages. Returns (replies by id, raw stdout, stderr)."""
    handshake = [
        {
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {
                "protocolVersion": types.LATEST_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "1"},
            },
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
    ]
    expected = {
        m["id"] for m in handshake + messages if isinstance(m, dict) and "id" in m
    }
    process = subprocess.Popen(
        [sys.executable, "-m", module],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=ROOT,
    )
    watchdog = threading.Timer(30, process.kill)
    watchdog.start()

    # Stdin stays open until every reply is in. Closing it early is a shutdown
    # signal and would cancel requests that are still running.
    for message in handshake + messages:
        process.stdin.write((message if isinstance(message, str) else json.dumps(message)) + "\n")
    process.stdin.flush()

    replies, raw = {}, []
    while not expected <= set(replies):
        line = process.stdout.readline()
        if not line:
            break
        raw.append(line)
        message = json.loads(line)  # raises if anything other than JSON got out
        assert message["jsonrpc"] == "2.0"
        replies[message.get("id")] = message

    process.stdin.close()
    raw.append(process.stdout.read())
    stderr = process.stderr.read()
    process.wait()
    watchdog.cancel()
    return replies, "".join(raw), stderr


def call(request_id, name, arguments):
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }


def single_call(name, arguments):
    replies, _, _ = run_session([call(1, name, arguments)])
    return replies[1]


def test_tools_list_advertises_the_schema_that_is_enforced():
    replies, _, _ = run_session([{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}])
    tools = {t["name"]: t for t in replies[1]["result"]["tools"]}
    assert set(tools) == {"get_customer_record", "trigger_refund"}

    schema = tools["trigger_refund"]["inputSchema"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"customer_id", "amount", "reason"}
    assert schema["properties"]["amount"]["exclusiveMinimum"] == 0
    assert schema["properties"]["reason"]["minLength"] == 10
    assert schema["properties"]["customer_id"]["pattern"] == "^CUST-[0-9]{5}$"


def test_get_customer_record():
    reply = single_call("get_customer_record", {"customer_id": "CUST-10001"})
    assert reply["result"]["isError"] is False
    record = json.loads(reply["result"]["content"][0]["text"])
    assert record["name"] == "Ada Lovelace"


def test_refund_reduces_balance_across_calls():
    replies, _, _ = run_session(
        [
            call(1, "trigger_refund", {"customer_id": "CUST-10002", "amount": 25.5, "reason": VALID_REASON}),
            call(2, "get_customer_record", {"customer_id": "CUST-10002"}),
        ]
    )
    refund = json.loads(replies[1]["result"]["content"][0]["text"])
    assert refund["status"] == "processed"
    record = json.loads(replies[2]["result"]["content"][0]["text"])
    assert record["refundable_balance"] == 50.0
    assert len(record["refunds"]) == 1


@pytest.mark.parametrize(
    "customer_id",
    ["CUST-1234", "CUST-123456", "cust-12345", "CUST12345", "CUST-1234a", " CUST-12345",
     "CUST-12345\n", "CUST-١٢٣٤٥", "", 12345, None, ["CUST-12345"]],
)
def test_bad_customer_id_is_a_protocol_error(customer_id):
    reply = single_call("get_customer_record", {"customer_id": customer_id})
    assert reply["error"]["code"] == types.INVALID_PARAMS
    assert reply["error"]["data"]["errors"][0]["field"] == "customer_id"


@pytest.mark.parametrize(
    "amount", [0, -5, -0.01, "10", "abc", None, True, [10], 10.555, 1e400, -1e400]
)
def test_bad_amount_is_a_protocol_error(amount):
    args = {"customer_id": "CUST-10001", "amount": amount, "reason": VALID_REASON}
    reply = single_call("trigger_refund", args)
    assert reply["error"]["code"] == types.INVALID_PARAMS
    assert reply["error"]["data"]["errors"][0]["field"] == "amount"


@pytest.mark.parametrize("reason", ["short", "123456789", " " * 30, "  short    ", "", None, 1234567890, "x" * 501])
def test_bad_reason_is_a_protocol_error(reason):
    args = {"customer_id": "CUST-10001", "amount": 10.0, "reason": reason}
    reply = single_call("trigger_refund", args)
    assert reply["error"]["code"] == types.INVALID_PARAMS
    assert reply["error"]["data"]["errors"][0]["field"] == "reason"


def test_integer_amount_is_accepted():
    # JSON has one number type, so 10 and 10.0 are the same value on the wire.
    args = {"customer_id": "CUST-10001", "amount": 10, "reason": VALID_REASON}
    assert "result" in single_call("trigger_refund", args)


@pytest.mark.parametrize(
    "args",
    [
        {},
        {"customer_id": "CUST-10001", "extra": "field"},
        {"customerId": "CUST-10001"},
    ],
)
def test_missing_and_unknown_fields_are_rejected(args):
    reply = single_call("get_customer_record", args)
    assert reply["error"]["code"] == types.INVALID_PARAMS


def test_rejected_values_are_not_echoed_back():
    secret = "4111111111111111"
    reply = single_call("get_customer_record", {"customer_id": secret})
    assert secret not in json.dumps(reply)


def test_unknown_tool_is_a_protocol_error():
    reply = single_call("delete_everything", {})
    assert reply["error"]["code"] == types.INVALID_PARAMS


def test_unknown_method():
    replies, _, _ = run_session([{"jsonrpc": "2.0", "id": 1, "method": "tools/explode"}])
    assert replies[1]["error"]["code"] == types.METHOD_NOT_FOUND


def test_business_failures_come_back_as_tool_results():
    missing = single_call("get_customer_record", {"customer_id": "CUST-99999"})
    assert missing["result"]["isError"] is True
    assert "customer_not_found" in missing["result"]["content"][0]["text"]

    args = {"customer_id": "CUST-10002", "amount": 1000.0, "reason": VALID_REASON}
    too_much = single_call("trigger_refund", args)
    assert too_much["result"]["isError"] is True
    assert "refund_exceeds_balance" in too_much["result"]["content"][0]["text"]


def test_server_survives_garbage_on_stdin():
    replies, _, _ = run_session(
        [
            "this is not json",
            '{"jsonrpc": "2.0", "id": 5',
            call(1, "get_customer_record", {"customer_id": "CUST-10001"}),
        ]
    )
    assert "result" in replies[1]


def test_logs_go_to_stderr():
    _, stdout, stderr = run_session([call(1, "get_customer_record", {"customer_id": "CUST-10001"})])
    assert "starting on stdio" in stderr
    assert "starting on stdio" not in stdout


def test_stray_output_cannot_reach_stdout():
    args = {"customer_id": "CUST-10001"}
    replies, stdout, stderr = run_session(
        [call(1, "get_customer_record", args)],
        module="task1_mcp_server.tests.noisy_server",
    )
    assert "result" in replies[1]  # run_session already parsed every stdout line
    assert "debug:" not in stdout
    assert "debug: print() from a handler" in stderr
    assert "debug: child process output" in stderr
