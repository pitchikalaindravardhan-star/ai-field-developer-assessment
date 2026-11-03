"""Fake OpenAI-compatible provider that streams a canned answer in small chunks.

Run with:  uvicorn task3_pii_gateway.mock_llm:app --port 8001
"""

import asyncio
import json
import os

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import StreamingResponse
from starlette.routing import Route

ANSWER = (
    "Sure. The customer is reachable at jane.doe@example.com and her SSN is "
    "123-45-6789. She paid with card 4111 1111 1111 1111 on 2024-01-15, "
    "order number 8842."
)
CHUNK_SIZE = int(os.environ.get("MOCK_CHUNK_SIZE", "4"))
DELAY = float(os.environ.get("MOCK_DELAY_SECONDS", "0.02"))


def _chunk(delta: dict, finish_reason: str | None = None) -> str:
    payload = {
        "id": "chatcmpl-mock",
        "object": "chat.completion.chunk",
        "model": "mock",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(payload)}\n\n"


async def _events():
    yield _chunk({"role": "assistant", "content": ""})
    for start in range(0, len(ANSWER), CHUNK_SIZE):
        await asyncio.sleep(DELAY)
        yield _chunk({"content": ANSWER[start : start + CHUNK_SIZE]})
    yield _chunk({}, "stop")
    yield "data: [DONE]\n\n"


async def chat_completions(request: Request) -> StreamingResponse:
    return StreamingResponse(_events(), media_type="text/event-stream")


app = Starlette(routes=[Route("/v1/chat/completions", chat_completions, methods=["POST"])])
