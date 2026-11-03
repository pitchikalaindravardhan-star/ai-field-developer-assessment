import asyncio
import multiprocessing
import sqlite3

import pytest

from task4_model_router.rate_limiter import SlidingWindowLimiter, retry_after_seconds, tenant_id


class FakeClock:
    def __init__(self):
        self.now = 1_000_000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
async def limiter(tmp_path, clock):
    limiter = SlidingWindowLimiter(str(tmp_path / "limits.db"), limit=50_000, clock=clock)
    yield limiter
    await limiter.close()


async def test_allows_up_to_the_limit_and_no_further(limiter):
    assert (await limiter.reserve("t", 30_000)).allowed
    assert (await limiter.reserve("t", 20_000)).allowed
    assert await limiter.usage("t") == 50_000
    assert not (await limiter.reserve("t", 1)).allowed


async def test_window_slides_rather_than_resets(limiter, clock):
    await limiter.reserve("t", 30_000)
    clock.advance(40)
    await limiter.reserve("t", 20_000)
    clock.advance(25)  # the first reservation is gone, the second isn't
    assert await limiter.usage("t") == 20_000
    assert (await limiter.reserve("t", 30_000)).allowed
    assert not (await limiter.reserve("t", 1)).allowed


async def test_eviction_boundary_is_exact(limiter, clock):
    await limiter.reserve("t", 50_000)
    clock.advance(59.999)
    assert not (await limiter.reserve("t", 1)).allowed
    clock.advance(0.001)
    assert (await limiter.reserve("t", 50_000)).allowed


async def test_no_double_burst_across_a_minute_boundary(limiter, clock):
    clock.now = 1_000_059.0  # one second before a fixed window would reset
    await limiter.reserve("t", 50_000)
    clock.advance(2)
    assert not (await limiter.reserve("t", 1)).allowed


async def test_tenants_do_not_share_a_budget(limiter):
    await limiter.reserve("a", 50_000)
    assert (await limiter.reserve("b", 50_000)).allowed


async def test_expired_rows_are_deleted_from_disk(limiter, clock):
    await limiter.reserve("t", 100)
    clock.advance(61)
    await limiter.reserve("t", 100)
    assert await limiter.row_count() == 1


async def test_idle_tenants_are_swept(limiter, clock):
    for tenant in ("a", "b", "c"):
        await limiter.reserve(tenant, 100)
    clock.advance(61)
    await limiter.reserve("d", 100)
    assert await limiter.row_count() == 1


async def test_retry_after_is_the_moment_room_appears(limiter, clock):
    await limiter.reserve("t", 30_000)
    clock.advance(10)
    await limiter.reserve("t", 20_000)
    decision = await limiter.reserve("t", 10_000)
    assert decision.retry_after_ms == 50_000

    clock.advance(50)
    assert (await limiter.reserve("t", 10_000)).allowed


async def test_retry_after_seconds_rounds_up():
    assert retry_after_seconds(1) == 1
    assert retry_after_seconds(1001) == 2


async def test_request_above_the_limit_is_flagged_as_never_fitting(limiter):
    decision = await limiter.reserve("t", 50_001)
    assert decision.too_large
    assert await limiter.usage("t") == 0


async def test_settle_corrects_the_estimate_both_ways(limiter):
    high = await limiter.reserve("t", 40_000)
    await limiter.settle(high.reservation_id, 10_000)
    assert await limiter.usage("t") == 10_000

    low = await limiter.reserve("t", 100)
    await limiter.settle(low.reservation_id, 45_000)
    assert await limiter.usage("t") == 55_000
    assert not (await limiter.reserve("t", 1)).allowed


async def test_release_returns_the_tokens(limiter):
    decision = await limiter.reserve("t", 50_000)
    await limiter.release(decision.reservation_id)
    assert (await limiter.reserve("t", 50_000)).allowed


async def test_concurrent_requests_cannot_overshoot(limiter):
    decisions = await asyncio.gather(*(limiter.reserve("t", 1_000) for _ in range(200)))
    assert sum(d.allowed for d in decisions) == 50
    assert await limiter.usage("t") == 50_000


def _hammer(path, results):
    async def run():
        limiter = SlidingWindowLimiter(path, limit=50_000)
        decisions = await asyncio.gather(*(limiter.reserve("shared", 1_000) for _ in range(40)))
        await limiter.close()
        return sum(d.allowed for d in decisions)

    results.put(asyncio.run(run()))


def test_limit_holds_across_worker_processes(tmp_path):
    path = str(tmp_path / "limits.db")
    SlidingWindowLimiter(path)  # create the schema before the race starts
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    workers = [context.Process(target=_hammer, args=(path, results)) for _ in range(4)]
    for worker in workers:
        worker.start()
    granted = sum(results.get(timeout=30) for _ in workers)
    for worker in workers:
        worker.join()
    assert granted == 50


async def test_state_is_on_disk_and_survives_a_restart(tmp_path, clock):
    path = str(tmp_path / "limits.db")
    first = SlidingWindowLimiter(path, clock=clock)
    await first.reserve("t", 50_000)
    await first.close()

    second = SlidingWindowLimiter(path, clock=clock)
    assert not (await second.reserve("t", 1)).allowed
    await second.close()


async def test_raw_api_keys_are_not_stored(tmp_path, clock):
    path = str(tmp_path / "limits.db")
    limiter = SlidingWindowLimiter(path, clock=clock)
    await limiter.reserve(tenant_id("sk-tenant-secret"), 100)
    await limiter.close()
    rows = sqlite3.connect(path).execute("SELECT tenant FROM token_usage").fetchall()
    assert "sk-tenant-secret" not in rows[0][0]
