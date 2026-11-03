import json

import httpx
import pytest
from asgi_lifespan import LifespanManager

from task3_pii_gateway import mock_llm
from task3_pii_gateway.gateway import create_app, redact_sse
from task3_pii_gateway.redactor import REPLACEMENT


def chunk(content=None, finish=None, index=0):
    delta = {} if content is None else {"content": content}
    return json.dumps({"id": "x", "choices": [{"index": index, "delta": delta, "finish_reason": finish}]})


def data_lines(*chunks):
    return [f"data: {c}" for c in chunks]


async def lines_from(items, then=None):
    for item in items:
        yield item
    if then:
        raise then


async def collect(lines):
    return [event async for event in redact_sse(lines)]


def parse(events):
    return [json.loads(e[6:]) for e in events if e != b"data: [DONE]\n\n"]


def text_of(events, index=0):
    return "".join(
        choice["delta"].get("content") or ""
        for payload in parse(events)
        for choice in payload.get("choices", [])
        if choice["index"] == index
    )


async def test_secret_split_across_deltas_is_redacted():
    lines = data_lines(chunk("write to ja"), chunk("ne@exam"), chunk("ple.com please"), chunk(finish="stop"))
    events = await collect(lines_from(lines + ["data: [DONE]"]))
    assert text_of(events) == f"write to {REPLACEMENT} please"
    assert events[-1] == b"data: [DONE]\n\n"


async def test_held_text_arrives_before_the_finish_marker():
    events = await collect(lines_from(data_lines(chunk("the end"), chunk(finish="stop"))))
    payloads = parse(events)
    assert payloads[-1]["choices"][0]["finish_reason"] == "stop"
    assert text_of(events) == "the end"


async def test_each_choice_has_its_own_buffer():
    lines = data_lines(
        chunk("a@b", index=0), chunk("c@d", index=1),
        chunk(".com ok", index=0), chunk(".org ok", index=1),
        chunk(finish="stop", index=0), chunk(finish="stop", index=1),
    )
    events = await collect(lines_from(lines))
    assert text_of(events, 0) == f"{REPLACEMENT} ok"
    assert text_of(events, 1) == f"{REPLACEMENT} ok"


async def test_length_cutoff_does_not_leak_a_partial_value():
    lines = data_lines(chunk("card 4111 1111 11"), chunk(finish="length"))
    assert text_of(await collect(lines_from(lines))) == f"card {REPLACEMENT}"


async def test_unparseable_chunk_is_dropped_not_forwarded():
    lines = ["data: {broken jane@example.com"] + data_lines(chunk("ok "), chunk(finish="stop"))
    events = await collect(lines_from(lines))
    assert b"jane" not in b"".join(events)


async def test_role_and_usage_chunks_survive():
    role = json.dumps({"id": "x", "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
    usage = json.dumps({"id": "x", "choices": [], "usage": {"total_tokens": 9}})
    events = await collect(lines_from(data_lines(role, chunk("hi "), chunk(finish="stop"), usage)))
    payloads = parse(events)
    assert payloads[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert payloads[-1]["usage"] == {"total_tokens": 9}


async def test_provider_failure_mid_stream():
    failure = httpx.ReadError("connection to 10.0.0.5 lost")
    lines = lines_from(data_lines(chunk("mail john@exam")), then=failure)
    events = await collect(lines)
    joined = b"".join(events).decode()
    assert "john" not in joined
    assert "10.0.0.5" not in joined
    assert parse(events)[-1]["error"]["type"] == "upstream_error"


async def test_provider_read_stops_when_the_client_goes_away():
    pulled = []

    async def endless():
        for number in range(1000):
            pulled.append(number)
            yield f"data: {chunk(f'word{number} ')}"

    stream = redact_sse(endless())
    await anext(stream)
    await stream.aclose()
    assert len(pulled) < 5


def client_for(transport):
    app = create_app("http://llm/v1", "key", transport=transport)
    return app


async def request_through(app, body=None):
    async with LifespanManager(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://gateway"
        ) as client:
            return await client.post(
                "/v1/chat/completions",
                json=body if body is not None else {"messages": [{"role": "user", "content": "hi"}]},
            )


async def test_end_to_end_against_mock_provider(monkeypatch):
    monkeypatch.setattr(mock_llm, "DELAY", 0)
    routes = httpx.ASGITransport(app=mock_llm.app)
    reply = await request_through(client_for(routes))
    assert reply.headers["content-type"].startswith("text/event-stream")
    assert reply.text.endswith("data: [DONE]\n\n")

    events = [e.encode() + b"\n\n" for e in reply.text.split("\n\n") if e]
    assert text_of(events) == (
        f"Sure. The customer is reachable at {REPLACEMENT} and her SSN is "
        f"{REPLACEMENT}. She paid with card {REPLACEMENT} on 2024-01-15, "
        "order number 8842."
    )


@pytest.mark.parametrize("status, expected", [(429, 429), (500, 502), (401, 502)])
async def test_provider_error_body_is_not_passed_on(status, expected):
    def reject(request):
        return httpx.Response(status, json={"error": "key sk-secret rejected for prompt: my SSN"})

    reply = await request_through(client_for(httpx.MockTransport(reject)))
    assert reply.status_code == expected
    assert "sk-secret" not in reply.text


async def test_provider_unreachable():
    def refuse(request):
        raise httpx.ConnectError("10.0.0.5 refused")

    reply = await request_through(client_for(httpx.MockTransport(refuse)))
    assert reply.status_code == 502
    assert "10.0.0.5" not in reply.text


async def test_provider_without_an_api_key(monkeypatch):
    monkeypatch.setattr(mock_llm, "DELAY", 0)
    app = create_app("http://llm/v1", "none", transport=httpx.ASGITransport(app=mock_llm.app))
    app.state.api_key = ""
    assert (await request_through(app)).status_code == 200


@pytest.mark.parametrize("body", [{}, {"messages": "hi"}, []])
async def test_bad_request(body):
    reply = await request_through(client_for(httpx.MockTransport(lambda r: httpx.Response(200))), body)
    assert reply.status_code == 400
