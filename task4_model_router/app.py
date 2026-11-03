"""HTTP front end for the model router.

Run with:
    GATEWAY_API_KEYS=tenant-a,tenant-b \
    PRIMARY_URL=http://127.0.0.1:8101/v1/chat/completions PRIMARY_API_KEY=... PRIMARY_MODEL=... \
    SECONDARY_URL=http://127.0.0.1:8102/v1/chat/completions SECONDARY_API_KEY=... SECONDARY_MODEL=... \
        uvicorn task4_model_router.app:create_app --factory --port 8100
"""

import asyncio
import logging
import math
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import uuid4

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from . import errors
from .errors import GatewayError, error_response
from .rate_limiter import SlidingWindowLimiter, retry_after_seconds, tenant_id
from .router import ModelRouter, Provider

log = logging.getLogger("llm-gateway")

DEFAULT_MAX_TOKENS = 1024


def estimate_tokens(payload: dict) -> int:
    """Rough upper estimate: about 4 characters per token, plus the reply budget.

    Close enough to reserve with, since settle() corrects it afterwards.
    """
    characters = 0
    for message in payload["messages"]:
        content = message.get("content") if isinstance(message, dict) else None
        characters += len(content) if isinstance(content, str) else len(str(content or ""))
    max_tokens = payload.get("max_tokens")
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        max_tokens = DEFAULT_MAX_TOKENS
    return math.ceil(characters / 4) + max_tokens


def _tenant_from(request: Request) -> str:
    scheme, _, key = request.headers.get("authorization", "").partition(" ")
    tenant = tenant_id(key.strip())
    if scheme.lower() != "bearer" or tenant not in request.app.state.tenants:
        raise errors.unauthorized()
    return tenant


async def _read_payload(request: Request) -> dict:
    try:
        payload = await request.json()
    except ValueError:
        raise errors.invalid_request("Request body must be valid JSON.") from None
    if not isinstance(payload, dict):
        raise errors.invalid_request("Request body must be a JSON object.")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise errors.invalid_request("messages must be a non-empty array.")
    if payload.get("stream"):
        raise errors.invalid_request("Streaming is not supported on this endpoint.")
    return payload


async def _complete(request: Request, request_id: str) -> Response:
    state = request.app.state
    tenant = _tenant_from(request)
    payload = await _read_payload(request)
    estimate = estimate_tokens(payload)

    decision = await state.limiter.reserve(tenant, estimate)
    if decision.too_large:
        raise errors.request_too_large()
    if not decision.allowed:
        raise errors.rate_limited(retry_after_seconds(decision.retry_after_ms))

    settled = False
    try:
        completion = await state.router.complete(payload, request_id)
        settled = True
        if completion.total_tokens is not None:
            await state.limiter.settle(decision.reservation_id, completion.total_tokens)
    finally:
        if not settled:
            # All providers failed, or the client hung up and this task is being
            # cancelled. Shielded so a cancellation can't skip the release and
            # leave the tenant paying for tokens nobody used.
            await asyncio.shield(state.limiter.release(decision.reservation_id))

    return JSONResponse(
        completion.body,
        headers={
            "X-Request-Id": request_id,
            "X-Gateway-Route": "primary" if completion.provider == state.primary_name else "fallback",
        },
    )


async def chat_completions(request: Request) -> Response:
    request_id = f"req_{uuid4().hex}"
    try:
        return await _complete(request, request_id)
    except GatewayError as error:
        return error_response(error, request_id)
    except Exception:
        # The traceback stays here. The client gets the generic payload and an id
        # that finds this log line.
        log.exception("request=%s unhandled failure", request_id)
        return error_response(errors.internal_error(), request_id)


def provider_from_env(prefix: str) -> Provider:
    return Provider(
        name=prefix.lower(),
        url=os.environ[f"{prefix}_URL"],
        api_key=os.environ[f"{prefix}_API_KEY"],
        model=os.environ[f"{prefix}_MODEL"],
    )


def create_app(
    primary: Provider | None = None,
    secondary: Provider | None = None,
    api_keys: list[str] | None = None,
    db_path: str | None = None,
    limit: int = 50_000,
    transport: httpx.AsyncBaseTransport | None = None,
    clock=None,
) -> Starlette:
    primary = primary or provider_from_env("PRIMARY")
    secondary = secondary or provider_from_env("SECONDARY")
    keys = api_keys if api_keys is not None else os.environ["GATEWAY_API_KEYS"].split(",")
    path = db_path or os.environ.get("RATE_LIMIT_DB", "rate_limits.db")

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        options = {"clock": clock} if clock else {}
        app.state.limiter = SlidingWindowLimiter(path, limit=limit, **options)
        async with httpx.AsyncClient(transport=transport) as client:
            app.state.router = ModelRouter(client, primary, secondary)
            yield
        await app.state.limiter.close()

    app = Starlette(
        routes=[Route("/v1/chat/completions", chat_completions, methods=["POST"])],
        lifespan=lifespan,
    )
    # Unknown keys must be refused. If any string got its own bucket, a client
    # could dodge the limit by changing its key on every request.
    app.state.tenants = frozenset(tenant_id(key.strip()) for key in keys if key.strip())
    app.state.primary_name = primary.name
    return app
