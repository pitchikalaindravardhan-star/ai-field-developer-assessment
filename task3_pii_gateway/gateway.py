"""LLM gateway endpoint that streams responses with PII removed.

Speaks the OpenAI chat completions streaming format, so it works with any
compatible provider. Run with:

    LLM_BASE_URL=https://api.openai.com/v1 LLM_API_KEY=... \
        uvicorn task3_pii_gateway.gateway:create_app --factory --port 8000

The stream is a chain of async generators with no queue in between, so one chunk
is pulled from the provider only when the client is ready for one. A slow client
slows the provider read instead of filling memory, and when a client disconnects
the generator closes, which closes the provider connection.
"""

import json
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from .redactor import StreamRedactor

log = logging.getLogger("pii-gateway")

DONE_EVENT = b"data: [DONE]\n\n"


def _sse(payload: dict) -> bytes:
    return b"data: " + json.dumps(payload, ensure_ascii=False).encode() + b"\n\n"


def _text_chunk(template: dict, index: int, text: str) -> dict:
    """A chunk carrying held-back text, in the same shape as the provider's chunks."""
    return {
        **{k: v for k, v in template.items() if k not in ("choices", "usage")},
        "choices": [{"index": index, "delta": {"content": text}, "finish_reason": None}],
    }


class ChunkRedactor:
    """Redacts provider chunks, tracking one StreamRedactor per choice index."""

    def __init__(self) -> None:
        self._redactors: dict[int, StreamRedactor] = {}
        self._last_chunk: dict = {}

    def process(self, chunk: dict) -> list[bytes]:
        self._last_chunk = chunk
        events = []
        kept_choices = []
        for choice in chunk.get("choices") or []:
            index = choice.get("index", 0)
            redactor = self._redactors.setdefault(index, StreamRedactor())
            delta = choice.get("delta") or {}
            finish_reason = choice.get("finish_reason")

            content = delta.get("content")
            if isinstance(content, str) and content:
                delta["content"] = redactor.feed(content)
                if not delta["content"] and len(delta) == 1 and not finish_reason:
                    continue  # all of it is being held, nothing to send yet

            if finish_reason:
                # Held text has to arrive before the finish marker.
                remainder = redactor.flush(truncated=finish_reason == "length")
                if remainder:
                    events.append(_sse(_text_chunk(chunk, index, remainder)))
            kept_choices.append(choice)

        if kept_choices or not chunk.get("choices"):
            events.append(_sse({**chunk, "choices": kept_choices}))
        return events

    def finish(self, truncated: bool = False) -> list[bytes]:
        """Releases anything still held. A no-op if every choice sent a finish_reason."""
        events = []
        for index, redactor in self._redactors.items():
            remainder = redactor.flush(truncated=truncated)
            if remainder:
                events.append(_sse(_text_chunk(self._last_chunk, index, remainder)))
        return events


async def redact_sse(lines: AsyncIterator[str]) -> AsyncIterator[bytes]:
    """Turns the provider's SSE lines into redacted SSE events."""
    redactor = ChunkRedactor()
    try:
        async for line in lines:
            if not line.startswith("data:"):
                continue  # blank separators and keep-alive comments
            data = line[5:].strip()
            if data == "[DONE]":
                for event in redactor.finish():
                    yield event
                yield DONE_EVENT
                return
            try:
                chunk = json.loads(data)
            except ValueError:
                # Can't inspect it, so it doesn't go out.
                log.warning("dropped an unparseable chunk from the provider")
                continue
            for event in redactor.process(chunk):
                yield event

        # Stream ended without [DONE], so the provider cut out.
        for event in redactor.finish(truncated=True):
            yield event
    except httpx.HTTPError as exc:
        log.warning("provider stream failed: %s", exc)
        for event in redactor.finish(truncated=True):
            yield event
        yield _sse({"error": {"type": "upstream_error", "message": "The provider stream failed"}})


async def chat_completions(request: Request) -> Response:
    try:
        body = await request.json()
    except ValueError:
        body = None
    if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
        return JSONResponse(
            {"error": {"type": "invalid_request", "message": "messages is required"}}, 400
        )

    settings = request.app.state
    # Local providers often run without a key, and an empty Bearer is an illegal header.
    headers = {"Authorization": f"Bearer {settings.api_key}"} if settings.api_key else {}
    upstream_request = settings.client.build_request(
        "POST",
        f"{settings.base_url}/chat/completions",
        json={**body, "stream": True},
        headers=headers,
    )
    try:
        upstream = await settings.client.send(upstream_request, stream=True)
    except httpx.HTTPError as exc:
        log.warning("provider unreachable: %s", exc)
        return JSONResponse(
            {"error": {"type": "upstream_error", "message": "The provider is unavailable"}}, 502
        )

    if upstream.status_code != 200:
        # The body is dropped unread. Providers can echo the prompt in errors.
        status = upstream.status_code
        await upstream.aclose()
        log.warning("provider returned %s", status)
        return JSONResponse(
            {"error": {"type": "upstream_error", "message": "The provider rejected the request"}},
            429 if status == 429 else 502,
        )

    async def stream() -> AsyncIterator[bytes]:
        try:
            async for event in redact_sse(upstream.aiter_lines()):
                yield event
        finally:
            await upstream.aclose()

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def create_app(
    base_url: str | None = None,
    api_key: str | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> Starlette:
    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        # read=60 is the gap allowed between chunks, not the total stream time.
        timeout = httpx.Timeout(10.0, read=60.0)
        async with httpx.AsyncClient(transport=transport, timeout=timeout) as client:
            app.state.client = client
            yield

    app = Starlette(
        routes=[Route("/v1/chat/completions", chat_completions, methods=["POST"])],
        lifespan=lifespan,
    )
    app.state.base_url = (base_url or os.environ["LLM_BASE_URL"]).rstrip("/")
    app.state.api_key = api_key or os.environ.get("LLM_API_KEY", "")
    return app
