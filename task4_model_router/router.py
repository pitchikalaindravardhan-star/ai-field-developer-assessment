"""Model routing with failover from a primary to a backup provider."""

import asyncio
import logging
from dataclasses import dataclass

import httpx

from . import errors

log = logging.getLogger("model-router")

DEFAULT_TIMEOUT_MS = 3000
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


@dataclass(frozen=True)
class Provider:
    name: str
    url: str
    api_key: str
    model: str
    timeout_ms: int = DEFAULT_TIMEOUT_MS


@dataclass(frozen=True)
class Completion:
    body: dict
    provider: str
    total_tokens: int | None


class _Retryable(Exception):
    """This provider failed in a way the next one might not."""

    def __init__(self, reason: str, timed_out: bool = False):
        super().__init__(reason)
        self.timed_out = timed_out


class ModelRouter:
    def __init__(self, client: httpx.AsyncClient, primary: Provider, secondary: Provider) -> None:
        self._client = client
        self._providers = (primary, secondary)

    async def complete(self, payload: dict, request_id: str) -> Completion:
        last_failure = None
        for provider in self._providers:
            try:
                return await self._call(provider, payload, request_id)
            except _Retryable as failure:
                # Details stay in the logs, tied to the request id the client got.
                log.warning("request=%s provider=%s failed: %s", request_id, provider.name, failure)
                last_failure = failure

        if last_failure.timed_out:
            raise errors.gateway_timeout()
        raise errors.unavailable()

    async def _call(self, provider: Provider, payload: dict, request_id: str) -> Completion:
        try:
            # httpx timeouts apply per socket operation, so a provider dripping
            # one byte a second would never trip them. This bounds the whole
            # exchange, and it cancels the request, which closes the connection.
            # A late primary answer therefore can't show up after failover and
            # get counted a second time.
            async with asyncio.timeout(provider.timeout_ms / 1000):
                response = await self._client.post(
                    provider.url,
                    json={**payload, "model": provider.model},
                    headers={
                        "Authorization": f"Bearer {provider.api_key}",
                        "X-Request-Id": request_id,
                    },
                    timeout=None,
                )
        except TimeoutError:
            raise _Retryable(f"no answer in {provider.timeout_ms}ms", timed_out=True) from None
        except httpx.HTTPError as exc:
            raise _Retryable(f"transport error: {type(exc).__name__}: {exc}") from None

        if response.status_code in RETRYABLE_STATUS:
            raise _Retryable(f"status {response.status_code}: {response.text[:500]}")

        if response.status_code != 200:
            # A 4xx here means this request or our provider credentials are bad.
            # The backup would refuse too, so don't spend tokens finding out.
            log.error(
                "request=%s provider=%s status=%s body=%s",
                request_id, provider.name, response.status_code, response.text[:500],
            )
            raise errors.upstream_rejected()

        try:
            body = response.json()
        except ValueError:
            raise _Retryable("provider returned invalid JSON") from None
        if not isinstance(body, dict):
            raise _Retryable("provider returned a non-object body")

        usage = body.get("usage")
        total = usage.get("total_tokens") if isinstance(usage, dict) else None
        return Completion(body, provider.name, total if isinstance(total, int) else None)
