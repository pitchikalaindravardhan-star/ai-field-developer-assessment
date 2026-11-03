import json
import time

import httpx
import jwt
import pytest
from asgi_lifespan import LifespanManager

from task2_mcp_gateway import mock_downstream
from task2_mcp_gateway.auth import issue_token
from task2_mcp_gateway.gateway import create_app

SECRET = "test-secret-that-is-long-enough-for-hs256"
FUTURE = int(time.time()) + 3600


def token(role="viewer", secret=SECRET, exp=FUTURE):
    return {"Authorization": f"Bearer {issue_token(role, secret, exp)}"}


def tool_call(name, request_id=1):
    return {"jsonrpc": "2.0", "id": request_id, "method": "tools/call", "params": {"name": name}}


TOOLS_LIST = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}


@pytest.fixture
def received():
    mock_downstream.app.state.received.clear()
    return mock_downstream.app.state.received


@pytest.fixture
async def gateway(received):
    downstream = httpx.ASGITransport(app=mock_downstream.app)
    app = create_app(SECRET, "http://downstream/mcp", transport=downstream)
    async with LifespanManager(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway"
        ) as client:
            yield client


async def test_tools_list_passes_through_unchanged(gateway, received):
    reply = await gateway.post("/mcp", json=TOOLS_LIST, headers=token())
    assert reply.status_code == 200
    assert [t["name"] for t in reply.json()["result"]["tools"]] == ["get_status", "admin_reset_key"]
    assert received[0]["payload"] == TOOLS_LIST


async def test_viewer_can_call_normal_tool(gateway, received):
    reply = await gateway.post("/mcp", json=tool_call("get_status"), headers=token())
    assert reply.json()["result"]["content"][0]["text"] == "ran get_status"
    assert len(received) == 1


async def test_viewer_blocked_from_admin_tool_without_touching_downstream(gateway, received):
    reply = await gateway.post("/mcp", json=tool_call("admin_reset_key", 42), headers=token())
    assert reply.json() == {
        "jsonrpc": "2.0",
        "id": 42,
        "error": {"code": -32001, "message": "Unauthorized Tool Call"},
    }
    assert received == []


async def test_admin_can_call_admin_tool(gateway, received):
    reply = await gateway.post("/mcp", json=tool_call("admin_reset_key"), headers=token("admin"))
    assert reply.json()["result"]["content"][0]["text"] == "ran admin_reset_key"


@pytest.mark.parametrize("role", ["Admin", "ADMIN", "administrator", "guest", " admin"])
async def test_role_match_is_exact(gateway, received, role):
    reply = await gateway.post("/mcp", json=tool_call("admin_reset_key"), headers=token(role))
    assert reply.json()["error"]["code"] == -32001
    assert received == []


@pytest.mark.parametrize("name", ["ADMIN_reset", "Admin_reset", " admin_reset", "\tadmin_reset"])
async def test_prefix_tricks_are_blocked(gateway, received, name):
    reply = await gateway.post("/mcp", json=tool_call(name), headers=token())
    assert reply.json()["error"]["code"] == -32001
    assert received == []


async def test_escaped_tool_name_is_decoded_before_the_check(gateway, received):
    raw = '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"\\u0061dmin_reset"}}'
    reply = await gateway.post("/mcp", content=raw, headers=token())
    assert reply.json()["error"]["code"] == -32001
    assert received == []


async def test_admin_tool_call_without_id_is_still_blocked(gateway, received):
    message = tool_call("admin_reset_key")
    del message["id"]
    reply = await gateway.post("/mcp", json=message, headers=token())
    assert reply.json()["error"]["code"] == -32001
    assert received == []


async def test_duplicate_keys_are_rejected(gateway, received):
    raw = (
        '{"jsonrpc":"2.0","id":1,"method":"tools/call","method":"tools/list",'
        '"params":{"name":"admin_reset_key"}}'
    )
    reply = await gateway.post("/mcp", content=raw, headers=token())
    assert reply.status_code == 400
    assert reply.json()["error"]["code"] == -32700
    assert received == []


async def test_batch_is_split_into_allowed_and_denied(gateway, received):
    batch = [tool_call("get_status", 1), tool_call("admin_reset_key", 2)]
    reply = await gateway.post("/mcp", json=batch, headers=token())
    by_id = {m["id"]: m for m in reply.json()}
    assert by_id[1]["result"]["content"][0]["text"] == "ran get_status"
    assert by_id[2]["error"]["code"] == -32001
    assert [m["params"]["name"] for m in received[0]["payload"]] == ["get_status"]


async def test_batch_of_only_denied_calls_never_goes_downstream(gateway, received):
    batch = [tool_call("admin_reset_key", 1), tool_call("admin_other", 2)]
    reply = await gateway.post("/mcp", json=batch, headers=token())
    assert [m["error"]["code"] for m in reply.json()] == [-32001, -32001]
    assert received == []


@pytest.mark.parametrize(
    "params", [None, {}, {"name": None}, {"name": 123}, {"name": ""}, {"name": ["admin_x"]}, "admin_x"]
)
async def test_bad_params_are_rejected(gateway, received, params):
    message = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": params}
    reply = await gateway.post("/mcp", json=message, headers=token("admin"))
    assert reply.json()["error"]["code"] == -32602
    assert received == []


@pytest.mark.parametrize(
    "body, code",
    [
        ("{not json", -32700),
        ("[" * 200_000, -32700),
        ("", -32700),
        ("[]", -32600),
        ('"tools/list"', -32600),
        ("42", -32600),
        ('{"id": 1, "method": "tools/list"}', -32600),
        ('{"jsonrpc": "1.0", "id": 1, "method": "tools/list"}', -32600),
        ('{"jsonrpc": "2.0", "id": 1, "method": 7}', -32600),
    ],
)
async def test_malformed_payloads(gateway, received, body, code):
    reply = await gateway.post("/mcp", content=body, headers=token())
    assert reply.json()["error"]["code"] == code
    assert received == []


async def test_oversized_body(gateway, received):
    body = json.dumps({**TOOLS_LIST, "padding": "x" * (1024 * 1024 + 1)})
    reply = await gateway.post("/mcp", content=body, headers=token())
    assert reply.status_code == 413
    assert received == []


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Bearer"},
        {"Authorization": "Bearer not.a.jwt"},
        {"Authorization": "Basic dXNlcjpwYXNz"},
        token(secret="wrong-secret-that-is-long-enough-for-hs256"),
        token(exp=int(time.time()) - 10),
    ],
)
async def test_bad_tokens_get_401(gateway, received, headers):
    reply = await gateway.post("/mcp", json=TOOLS_LIST, headers=headers)
    assert reply.status_code == 401
    assert reply.headers["www-authenticate"] == "Bearer"
    assert received == []


async def test_alg_none_token_is_rejected(gateway, received):
    forged = jwt.encode({"role": "admin", "exp": FUTURE}, None, algorithm="none")
    reply = await gateway.post(
        "/mcp", json=tool_call("admin_reset_key"), headers={"Authorization": f"Bearer {forged}"}
    )
    assert reply.status_code == 401
    assert received == []


async def test_token_without_role_is_rejected(gateway, received):
    tok = jwt.encode({"exp": FUTURE}, SECRET, algorithm="HS256")
    reply = await gateway.post("/mcp", json=TOOLS_LIST, headers={"Authorization": f"Bearer {tok}"})
    assert reply.status_code == 401


async def test_client_token_is_not_passed_downstream(gateway, received):
    await gateway.post("/mcp", json=TOOLS_LIST, headers={**token(), "Mcp-Session-Id": "abc"})
    assert "authorization" not in received[0]["headers"]
    assert received[0]["headers"]["mcp-session-id"] == "abc"


async def test_downstream_failure_is_sanitized():
    def refuse(request):
        raise httpx.ConnectError("connect to 10.1.2.3:9001 refused")

    app = create_app(SECRET, "http://downstream/mcp", transport=httpx.MockTransport(refuse))
    async with LifespanManager(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway"
        ) as client:
            reply = await client.post("/mcp", json=TOOLS_LIST, headers=token())
    assert reply.status_code == 502
    assert reply.json()["error"]["code"] == -32603
    assert "10.1.2.3" not in reply.text
