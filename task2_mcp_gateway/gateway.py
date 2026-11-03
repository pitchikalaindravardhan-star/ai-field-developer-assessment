"""MCP gateway: a reverse proxy that authorizes tool calls by token role.

Run with:
    GATEWAY_JWT_SECRET=... DOWNSTREAM_MCP_URL=http://127.0.0.1:9001/mcp \
        uvicorn task2_mcp_gateway.gateway:create_app --factory --port 9000
"""

import json
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from .auth import InvalidToken, role_from_header
from .jsonrpc import (
    INTERNAL_ERROR,
    INVALID_REQUEST,
    PARSE_ERROR,
    UNAUTHENTICATED,
    check_message,
    error_response,
    parse,
)

log = logging.getLogger("mcp-gateway")

MAX_BODY_BYTES = 1024 * 1024

# Only what the MCP transport needs goes downstream. Authorization is left out on
# purpose: the client's token was issued for this gateway, not for the server
# behind it, and passing it along is the token passthrough the MCP spec forbids.
FORWARD_REQUEST_HEADERS = (
    "accept",
    "content-type",
    "last-event-id",
    "mcp-protocol-version",
    "mcp-session-id",
)
FORWARD_RESPONSE_HEADERS = ("cache-control", "content-type", "mcp-session-id")


def _downstream_headers(request: Request) -> dict[str, str]:
    return {k: request.headers[k] for k in FORWARD_REQUEST_HEADERS if k in request.headers}


async def _forward(client: httpx.AsyncClient, url: str, request: Request, body: bytes) -> Response:
    """Sends a request downstream and streams the reply, so SSE passes straight through."""
    upstream_request = client.build_request(
        request.method, url, content=body, headers=_downstream_headers(request)
    )
    try:
        upstream = await client.send(upstream_request, stream=True)
    except httpx.TimeoutException:
        log.warning("downstream timed out")
        return JSONResponse(error_response(None, INTERNAL_ERROR, "Downstream timeout"), 504)
    except httpx.HTTPError as exc:
        log.warning("downstream unreachable: %s", exc)
        return JSONResponse(error_response(None, INTERNAL_ERROR, "Downstream unavailable"), 502)

    headers = {k: upstream.headers[k] for k in FORWARD_RESPONSE_HEADERS if k in upstream.headers}
    return StreamingResponse(
        upstream.aiter_raw(),
        status_code=upstream.status_code,
        headers=headers,
        background=BackgroundTask(upstream.aclose),
    )


async def _handle_batch(
    client: httpx.AsyncClient, url: str, request: Request, batch: list, role: str
) -> Response:
    """Checks every message in a batch. Allowed ones go down, denied ones are answered here.

    Without this a viewer could wrap an admin_ call in an array and walk past a
    check that only looks at single objects.
    """
    if not batch:
        return JSONResponse(error_response(None, INVALID_REQUEST, "Invalid Request"), 400)

    replies, allowed = [], []
    for message in batch:
        denied = check_message(message, role)
        if denied:
            replies.append(denied)
        else:
            allowed.append(message)

    if allowed:
        try:
            upstream = await client.post(
                url, content=json.dumps(allowed), headers=_downstream_headers(request)
            )
            answered = upstream.json() if upstream.content else []
        except (httpx.HTTPError, ValueError) as exc:
            log.warning("downstream batch failed: %s", exc)
            return JSONResponse(error_response(None, INTERNAL_ERROR, "Downstream unavailable"), 502)
        replies.extend(answered if isinstance(answered, list) else [answered])

    return JSONResponse(replies) if replies else Response(status_code=202)


async def handle_mcp(request: Request) -> Response:
    settings = request.app.state
    try:
        role = role_from_header(request.headers.get("authorization"), settings.jwt_secret)
    except InvalidToken as exc:
        log.info("rejected token: %s", exc)
        return JSONResponse(
            error_response(None, UNAUTHENTICATED, "Authentication required"),
            401,
            headers={"WWW-Authenticate": "Bearer"},
        )

    client, url = settings.client, settings.downstream_url

    # GET opens the server's event stream and DELETE ends a session. Neither has
    # a JSON-RPC body to inspect, but both still needed the token above.
    if request.method != "POST":
        return await _forward(client, url, request, b"")

    body = await request.body()
    if len(body) > MAX_BODY_BYTES:
        return JSONResponse(error_response(None, INVALID_REQUEST, "Payload too large"), 413)

    try:
        payload = parse(body)
    except (ValueError, RecursionError):  # bad JSON, duplicate keys, bad UTF-8, deep nesting
        return JSONResponse(error_response(None, PARSE_ERROR, "Parse error"), 400)

    if isinstance(payload, list):
        return await _handle_batch(client, url, request, payload, role)

    denied = check_message(payload, role)
    if denied:
        log.info("denied method=%s role=%s", payload.get("method") if isinstance(payload, dict) else None, role)
        # HTTP 200: the transport worked and the failure is in the JSON-RPC body,
        # which is where MCP clients look for it.
        return JSONResponse(denied)

    return await _forward(client, url, request, body)


def create_app(
    jwt_secret: str | None = None,
    downstream_url: str | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> Starlette:
    secret = jwt_secret or os.environ["GATEWAY_JWT_SECRET"]
    url = downstream_url or os.environ["DOWNSTREAM_MCP_URL"]

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        # read=None: an SSE stream can sit quiet for as long as a tool runs.
        timeout = httpx.Timeout(10.0, read=None)
        async with httpx.AsyncClient(transport=transport, timeout=timeout) as client:
            app.state.client = client
            yield

    app = Starlette(
        routes=[Route("/mcp", handle_mcp, methods=["POST", "GET", "DELETE"])],
        lifespan=lifespan,
    )
    app.state.jwt_secret = secret
    app.state.downstream_url = url
    return app
