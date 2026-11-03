import asyncio
import json
import time

import httpx
import pytest
from asgi_lifespan import LifespanManager

from task4_model_router.app import create_app, estimate_tokens
from task4_model_router.router import Provider

PRIMARY = Provider("primary", "http://primary/v1/chat/completions", "pk-primary-secret", "model-a", 200)
SECONDARY = Provider("secondary", "http://secondary/v1/chat/completions", "pk-backup-secret", "model-b", 200)
KEY = {"Authorization": "Bearer tenant-a"}
BODY = {"messages": [{"role": "user", "content": "hello"}], "max_tokens": 100}
INTERNAL = (
    "Traceback (most recent call last):\n  File \"/srv/app/inference.py\", line 88\n"
    "psycopg2.OperationalError: could not connect to 10.20.30.40:5432 password=hunter2"
)


def completion(tokens=50):
    return {"choices": [{"message": {"content": "hi"}}], "usage": {"total_tokens": tokens}}


class Upstreams:
    """Scripted providers. Each behaviour is a status code, an exception or a delay."""

    def __init__(self, primary=200, secondary=200):
        self.behaviour = {"primary": primary, "secondary": secondary}
        self.calls = []
        self.cancelled = []

    async def __call__(self, request):
        name = request.url.host
        self.calls.append((name, json.loads(request.content)))
        behaviour = self.behaviour[name]
        if isinstance(behaviour, Exception):
            raise behaviour
        if isinstance(behaviour, float):
            try:
                await asyncio.sleep(behaviour)
            except asyncio.CancelledError:
                self.cancelled.append(name)
                raise
            return httpx.Response(200, json=completion())
        if behaviour == 200:
            return httpx.Response(200, json=completion())
        return httpx.Response(behaviour, text=INTERNAL)

    @property
    def called(self):
        return [name for name, _ in self.calls]


@pytest.fixture
def build(tmp_path):
    def _build(upstreams, primary=PRIMARY, secondary=SECONDARY, limit=50_000):
        return create_app(
            primary, secondary, ["tenant-a", "tenant-b"], str(tmp_path / "limits.db"),
            limit=limit, transport=httpx.MockTransport(upstreams),
        )
    return _build


class Gateway:
    def __init__(self, app):
        self.app = app

    async def __aenter__(self):
        self.manager = LifespanManager(self.app)
        await self.manager.__aenter__()
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url="http://gateway"
        )
        return self

    async def __aexit__(self, *exc):
        await self.client.aclose()
        await self.manager.__aexit__(*exc)

    async def post(self, body=BODY, headers=KEY):
        return await self.client.post("/v1/chat/completions", json=body, headers=headers)


def assert_sanitized(reply):
    assert set(reply.json()) == {"error"}
    assert set(reply.json()["error"]) == {"code", "message", "request_id"}
    for leak in ("Traceback", "inference.py", "psycopg2", "10.20.30.40", "hunter2",
                 "pk-primary-secret", "pk-backup-secret", "primary/", "secondary/"):
        assert leak not in reply.text


async def test_primary_success_does_not_touch_backup(build):
    upstreams = Upstreams()
    async with Gateway(build(upstreams)) as gateway:
        reply = await gateway.post()
    assert reply.status_code == 200
    assert upstreams.called == ["primary"]
    assert reply.headers["x-gateway-route"] == "primary"
    assert upstreams.calls[0][1]["model"] == "model-a"


async def test_429_fails_over(build):
    upstreams = Upstreams(primary=429)
    async with Gateway(build(upstreams)) as gateway:
        reply = await gateway.post()
    assert reply.status_code == 200
    assert upstreams.called == ["primary", "secondary"]
    assert upstreams.calls[1][1]["model"] == "model-b"
    assert reply.headers["x-gateway-route"] == "fallback"


async def test_timeout_fails_over_and_cancels_the_primary(build):
    upstreams = Upstreams(primary=30.0)
    async with Gateway(build(upstreams)) as gateway:
        started = time.perf_counter()
        reply = await gateway.post()
        elapsed = time.perf_counter() - started
    assert reply.status_code == 200
    assert upstreams.called == ["primary", "secondary"]
    assert upstreams.cancelled == ["primary"]
    assert 0.2 <= elapsed < 0.5


async def test_default_timeout_is_3000ms(build):
    primary = Provider("primary", PRIMARY.url, "k", "model-a")
    assert primary.timeout_ms == 3000
    upstreams = Upstreams(primary=30.0)
    async with Gateway(build(upstreams, primary=primary)) as gateway:
        started = time.perf_counter()
        reply = await gateway.post()
        elapsed = time.perf_counter() - started
    assert reply.status_code == 200
    assert 2.95 <= elapsed < 3.4


async def test_slow_but_in_time_primary_is_not_abandoned(build):
    upstreams = Upstreams(primary=0.05)
    async with Gateway(build(upstreams)) as gateway:
        reply = await gateway.post()
    assert reply.headers["x-gateway-route"] == "primary"
    assert upstreams.called == ["primary"]


@pytest.mark.parametrize("failure", [500, 502, 503, 504, httpx.ConnectError("refused"), httpx.ReadError("reset")])
async def test_other_outages_also_fail_over(build, failure):
    upstreams = Upstreams(primary=failure)
    async with Gateway(build(upstreams)) as gateway:
        reply = await gateway.post()
    assert reply.status_code == 200
    assert upstreams.called == ["primary", "secondary"]


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
async def test_client_side_rejections_do_not_fail_over(build, status):
    upstreams = Upstreams(primary=status)
    async with Gateway(build(upstreams)) as gateway:
        reply = await gateway.post()
    assert reply.status_code == 502
    assert reply.json()["error"]["code"] == "upstream_error"
    assert upstreams.called == ["primary"]
    assert_sanitized(reply)


async def test_both_down_gives_sanitized_503(build):
    upstreams = Upstreams(primary=429, secondary=500)
    async with Gateway(build(upstreams)) as gateway:
        reply = await gateway.post()
    assert reply.status_code == 503
    assert reply.json()["error"]["code"] == "service_unavailable"
    assert_sanitized(reply)


async def test_both_timing_out_gives_504(build):
    upstreams = Upstreams(primary=30.0, secondary=30.0)
    async with Gateway(build(upstreams)) as gateway:
        reply = await gateway.post()
    assert reply.status_code == 504
    assert_sanitized(reply)


async def test_transport_error_text_does_not_leak(build):
    upstreams = Upstreams(primary=httpx.ConnectError("10.20.30.40:5432"), secondary=httpx.ConnectError("10.20.30.40"))
    async with Gateway(build(upstreams)) as gateway:
        assert_sanitized(await gateway.post())


async def test_invalid_json_from_primary_fails_over(build):
    async def handler(request):
        if request.url.host == "primary":
            return httpx.Response(200, text="<html>bad gateway</html>")
        return httpx.Response(200, json=completion())

    async with Gateway(build(handler)) as gateway:
        reply = await gateway.post()
    assert reply.headers["x-gateway-route"] == "fallback"


async def test_unexpected_exception_is_sanitized(build, monkeypatch):
    app = build(Upstreams())
    async with Gateway(app) as gateway:
        async def explode(*args):
            raise RuntimeError(INTERNAL)

        monkeypatch.setattr(app.state.router, "complete", explode)
        reply = await gateway.post()
    assert reply.status_code == 500
    assert_sanitized(reply)


async def test_usage_is_settled_to_the_real_number(build):
    app = build(Upstreams())
    async with Gateway(app) as gateway:
        await gateway.post()
        rows = await app.state.limiter._run(
            lambda: app.state.limiter._db.execute("SELECT tokens FROM token_usage").fetchall()
        )
    assert rows == [(50,)]


async def test_failed_request_costs_nothing(build):
    app = build(Upstreams(primary=500, secondary=500))
    async with Gateway(app) as gateway:
        await gateway.post()
        assert await app.state.limiter.row_count() == 0


async def test_failover_charges_once(build):
    app = build(Upstreams(primary=429))
    async with Gateway(app) as gateway:
        await gateway.post()
        assert await app.state.limiter.row_count() == 1


async def test_client_disconnect_releases_the_reservation(tmp_path):
    from task4_model_router import app as app_module

    upstreams = Upstreams(primary=30.0)
    app = create_app(
        Provider("primary", PRIMARY.url, "k", "a", 30_000), SECONDARY, ["tenant-a"],
        str(tmp_path / "limits.db"), transport=httpx.MockTransport(upstreams),
    )

    async with LifespanManager(app):
        request = type("R", (), {"app": app, "headers": {"authorization": "Bearer tenant-a"}})()

        async def body():
            return BODY

        request.json = body
        task = asyncio.create_task(app_module._complete(request, "req_test"))
        while not upstreams.calls:
            await asyncio.sleep(0.01)
        assert await app.state.limiter.row_count() == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.05)
        assert await app.state.limiter.row_count() == 0
        assert upstreams.cancelled == ["primary"]


async def test_rate_limit_returns_standard_429_before_calling_anyone(build):
    upstreams = Upstreams()
    async with Gateway(build(upstreams, limit=160)) as gateway:
        assert (await gateway.post()).status_code == 200  # settles to 50
        assert (await gateway.post()).status_code == 200  # settles to 50
        calls_before = len(upstreams.calls)
        reply = await gateway.post()
    assert reply.status_code == 429
    assert reply.json()["error"]["code"] == "rate_limit_exceeded"
    assert int(reply.headers["retry-after"]) >= 1
    assert len(upstreams.calls) == calls_before
    assert_sanitized(reply)


async def test_tenants_are_limited_separately(build):
    async with Gateway(build(Upstreams(), limit=160)) as gateway:
        await gateway.post()
        await gateway.post(headers={"Authorization": "Bearer tenant-b"})
        assert (await gateway.post(headers=KEY)).status_code == 200  # a settles to 50
        assert (await gateway.post(headers=KEY)).status_code == 429


async def test_oversized_request(build):
    upstreams = Upstreams()
    async with Gateway(build(upstreams, limit=10)) as gateway:
        reply = await gateway.post()
    assert reply.status_code == 413
    assert upstreams.calls == []


async def test_concurrent_requests_respect_the_limit(build):
    upstreams = Upstreams()
    # Estimate is 102 tokens per request. Settling lowers it to 50, so slow the
    # providers enough that all the reservations overlap.
    upstreams.behaviour["primary"] = 0.05
    async with Gateway(build(upstreams, limit=1020)) as gateway:
        replies = await asyncio.gather(*(gateway.post() for _ in range(40)))
    assert sum(r.status_code == 200 for r in replies) == 10
    assert sum(r.status_code == 429 for r in replies) == 30


@pytest.mark.parametrize(
    "headers", [{}, {"Authorization": "Bearer unknown"}, {"Authorization": "Basic tenant-a"}]
)
async def test_unknown_keys_are_refused(build, headers):
    upstreams = Upstreams()
    async with Gateway(build(upstreams)) as gateway:
        reply = await gateway.post(headers=headers)
    assert reply.status_code == 401
    assert upstreams.calls == []


@pytest.mark.parametrize(
    "body", [{}, {"messages": []}, {"messages": "hi"}, [], {"messages": [{}], "stream": True}]
)
async def test_bad_bodies(build, body):
    upstreams = Upstreams()
    async with Gateway(build(upstreams)) as gateway:
        reply = await gateway.post(body=body)
    assert reply.status_code == 400
    assert reply.json()["error"]["code"] == "invalid_request"
    assert upstreams.calls == []


def test_estimate_counts_prompt_and_reply_budget():
    assert estimate_tokens({"messages": [{"content": "x" * 400}], "max_tokens": 100}) == 200
    assert estimate_tokens({"messages": [{"content": "x" * 400}]}) == 1124
    multipart = {"messages": [{"content": [{"type": "text", "text": "x" * 40}]}], "max_tokens": 1}
    assert estimate_tokens(multipart) > 10
