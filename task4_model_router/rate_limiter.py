"""Token-aware sliding window rate limiter backed by an on-disk SQLite file.

Every admitted request is one row: (tenant, timestamp, tokens). A tenant's usage
is the sum of its rows inside the last window. That gives a true sliding window
with no fixed-bucket reset where a tenant could burst twice the limit.

Tokens are reserved up front from an estimate, then corrected once the provider
reports real usage. Checking only after the response would let a burst of
concurrent requests all pass while the counter still reads zero.

The check and the insert run inside BEGIN IMMEDIATE, which takes SQLite's write
lock before reading. That makes the pair atomic across coroutines, threads and
separate worker processes sharing the file.

sqlite3 is blocking, so all database work runs on one dedicated thread and the
event loop never waits on disk.
"""

import asyncio
import hashlib
import math
import sqlite3
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

SCHEMA = """
CREATE TABLE IF NOT EXISTS token_usage (
    id        INTEGER PRIMARY KEY,
    tenant    TEXT    NOT NULL,
    ts_ms     INTEGER NOT NULL,
    tokens    INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_usage_tenant_ts ON token_usage (tenant, ts_ms);
CREATE INDEX IF NOT EXISTS idx_usage_ts ON token_usage (ts_ms);
"""


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reservation_id: int | None = None
    retry_after_ms: int = 0
    too_large: bool = False  # can never fit, so retrying is pointless


def tenant_id(api_key: str) -> str:
    """Raw API keys don't belong on disk."""
    return hashlib.sha256(api_key.encode()).hexdigest()


class SlidingWindowLimiter:
    def __init__(
        self,
        db_path: str,
        limit: int = 50_000,
        window_ms: int = 60_000,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.limit = limit
        self.window_ms = window_ms
        self._clock = clock
        self._last_sweep_ms = 0
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rate-limiter")
        self._db = sqlite3.connect(
            db_path,
            timeout=5.0,  # busy timeout when another process holds the write lock
            isolation_level=None,  # transactions are managed by hand below
            check_same_thread=False,  # only ever used from the one executor thread
        )
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.executescript(SCHEMA)

    def _now_ms(self) -> int:
        # Wall clock, not monotonic: it has to mean the same thing across processes.
        return int(self._clock() * 1000)

    async def _run(self, fn, *args):
        return await asyncio.get_running_loop().run_in_executor(self._executor, fn, *args)

    async def reserve(self, tenant: str, tokens: int) -> Decision:
        return await self._run(self._reserve, tenant, tokens)

    async def settle(self, reservation_id: int, actual_tokens: int) -> None:
        """Replaces the estimate with what the provider really used."""
        await self._run(self._settle, reservation_id, actual_tokens)

    async def release(self, reservation_id: int) -> None:
        """Gives tokens back for a request that produced nothing."""
        await self._run(self._release, reservation_id)

    async def usage(self, tenant: str) -> int:
        return await self._run(self._usage, tenant)

    async def row_count(self) -> int:
        return await self._run(lambda: self._db.execute("SELECT COUNT(*) FROM token_usage").fetchone()[0])

    async def close(self) -> None:
        await self._run(self._db.close)
        self._executor.shutdown(wait=False)

    def _reserve(self, tenant: str, tokens: int) -> Decision:
        if tokens > self.limit:
            return Decision(allowed=False, too_large=True)

        now = self._now_ms()
        cutoff = now - self.window_ms
        db = self._db
        db.execute("BEGIN IMMEDIATE")
        try:
            # A row is out of the window once it is exactly window_ms old.
            db.execute("DELETE FROM token_usage WHERE tenant = ? AND ts_ms <= ?", (tenant, cutoff))
            self._sweep_idle_tenants(now, cutoff)

            used = db.execute(
                "SELECT COALESCE(SUM(tokens), 0) FROM token_usage WHERE tenant = ?", (tenant,)
            ).fetchone()[0]

            if used + tokens > self.limit:
                retry_after = self._ms_until_room(tenant, used + tokens - self.limit, now)
                db.execute("COMMIT")  # the eviction above is still worth keeping
                return Decision(allowed=False, retry_after_ms=retry_after)

            cursor = db.execute(
                "INSERT INTO token_usage (tenant, ts_ms, tokens) VALUES (?, ?, ?)",
                (tenant, now, tokens),
            )
            db.execute("COMMIT")
            return Decision(allowed=True, reservation_id=cursor.lastrowid)
        except BaseException:
            db.execute("ROLLBACK")
            raise

    def _sweep_idle_tenants(self, now: int, cutoff: int) -> None:
        # Per-tenant eviction only runs when that tenant makes a request, so a
        # tenant that goes quiet would keep its rows forever without this.
        if now - self._last_sweep_ms >= self.window_ms:
            self._db.execute("DELETE FROM token_usage WHERE ts_ms <= ?", (cutoff,))
            self._last_sweep_ms = now

    def _ms_until_room(self, tenant: str, overage: int, now: int) -> int:
        """Walks rows oldest first until enough tokens would have expired."""
        freed = 0
        rows = self._db.execute(
            "SELECT ts_ms, tokens FROM token_usage WHERE tenant = ? ORDER BY ts_ms", (tenant,)
        )
        for ts_ms, row_tokens in rows:
            freed += row_tokens
            if freed >= overage:
                return max(ts_ms + self.window_ms - now, 1)
        return self.window_ms

    def _settle(self, reservation_id: int, actual_tokens: int) -> None:
        self._db.execute(
            "UPDATE token_usage SET tokens = ? WHERE id = ?", (actual_tokens, reservation_id)
        )

    def _release(self, reservation_id: int) -> None:
        self._db.execute("DELETE FROM token_usage WHERE id = ?", (reservation_id,))

    def _usage(self, tenant: str) -> int:
        cutoff = self._now_ms() - self.window_ms
        return self._db.execute(
            "SELECT COALESCE(SUM(tokens), 0) FROM token_usage WHERE tenant = ? AND ts_ms > ?",
            (tenant, cutoff),
        ).fetchone()[0]


def retry_after_seconds(retry_after_ms: int) -> int:
    return max(1, math.ceil(retry_after_ms / 1000))
