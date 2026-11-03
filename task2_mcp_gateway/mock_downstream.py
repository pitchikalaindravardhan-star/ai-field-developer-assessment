"""Mock MCP server for the gateway to sit in front of.

Run with:  uvicorn task2_mcp_gateway.mock_downstream:app --port 9001
"""

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

TOOLS = [
    {"name": "get_status", "description": "Service status", "inputSchema": {"type": "object"}},
    {"name": "admin_reset_key", "description": "Rotate an API key", "inputSchema": {"type": "object"}},
]


def answer(message: dict) -> dict | None:
    if "id" not in message:
        return None
    if message.get("method") == "tools/list":
        result = {"tools": TOOLS}
    elif message.get("method") == "tools/call":
        text = f"ran {message['params']['name']}"
        result = {"content": [{"type": "text", "text": text}], "isError": False}
    else:
        result = {}
    return {"jsonrpc": "2.0", "id": message["id"], "result": result}


async def mcp(request: Request) -> Response:
    payload = await request.json()
    request.app.state.received.append(
        {"payload": payload, "headers": dict(request.headers)}
    )
    if isinstance(payload, list):
        return JSONResponse([r for r in map(answer, payload) if r])
    reply = answer(payload)
    return JSONResponse(reply) if reply else Response(status_code=202)


app = Starlette(routes=[Route("/mcp", mcp, methods=["POST"])])
app.state.received = []
