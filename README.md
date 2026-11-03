# AI Field Developer Assessment

Solutions to the four tasks, all in Python 3.13. Each task sits in its own folder and runs on its own.

| Task | Folder | What it is |
| --- | --- | --- |
| 1 | `task1_mcp_server` | MCP server over stdio with strict validation |
| 2 | `task2_mcp_gateway` | HTTP gateway that authorizes MCP tool calls by token role |
| 3 | `task3_pii_gateway` | LLM streaming proxy that redacts PII in real time |
| 4 | `task4_model_router` | Token rate limiter on SQLite plus model failover |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest
```

## Task 1: MCP server

```bash
python -m task1_mcp_server.server
```

Add it to an MCP client as a stdio server with that same command. `CUST-10001` and `CUST-10002` exist in the mock store.

- **Validation:** Pydantic models with `strict=True` and `extra="forbid"`. The schema in `tools/list` is generated from those models, so it can't disagree with what gets enforced.
- **Error mapping:** a bad request (unknown tool, invalid arguments) is a JSON-RPC error with code `-32602`. A valid request that fails on business rules (unknown customer, refund above balance) is a tool result with `isError: true`, as the MCP spec describes. I used the low-level `Server` because the higher-level `MCPServer` reports validation failures as `isError` results, which the task doesn't want.
- **Stdout:** logging is set to stderr. The SDK transport also moves the real stdout to a private descriptor while it runs. `noisy_server.py` proves this by calling `print()`, `os.write(1, ...)` and a child process from inside a handler, then checks that every stdout line is still valid JSON-RPC.
- **Assumptions:** `XXXXX` in `CUST-XXXXX` means five digits. Amounts are limited to two decimal places, and `reason` is measured after trimming whitespace.

## Task 2: MCP gateway

```bash
uvicorn task2_mcp_gateway.mock_downstream:app --port 9001

GATEWAY_JWT_SECRET=change-me-to-at-least-32-characters-long \
DOWNSTREAM_MCP_URL=http://127.0.0.1:9001/mcp \
uvicorn task2_mcp_gateway.gateway:create_app --factory --port 9000
```

Tokens are HS256 JWTs with a `role` claim and a required `exp`. `issue_token` in `auth.py` makes one for local testing.

The authorization rule is short. Most of the work is closing ways around it:

- **Batches:** a JSON-RPC array could hide an `admin_` call from a check that only reads objects, so each message is checked. Allowed ones go downstream and denied ones are answered by the gateway.
- **Duplicate keys:** `{"method": "tools/call", "method": "tools/list"}` can be read differently by the gateway and the server, so these are refused.
- **Name tricks:** `ADMIN_x` and ` admin_x` are normalized before the prefix check. JSON escapes like `admin_x` are decoded by the parser first.
- **Token passthrough:** the client's token is not sent downstream, as the MCP spec requires.
- **GET and DELETE:** these still need a valid token, and SSE responses are streamed through untouched.

`-32001` comes back with HTTP 200 because the transport worked and MCP clients read errors from the JSON-RPC body. A missing or invalid token gets HTTP 401.

`tools/list` is forwarded unchanged, as the task specifies, so viewers can still see admin tool names. In production I would filter that list by role too.

## Task 3: Streaming PII redaction

```bash
uvicorn task3_pii_gateway.mock_llm:app --port 8001

LLM_BASE_URL=http://127.0.0.1:8001/v1 \
uvicorn task3_pii_gateway.gateway:create_app --factory --port 8000

curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"messages": [{"role": "user", "content": "hi"}]}'
```

It works with any OpenAI-compatible provider through `LLM_BASE_URL` and `LLM_API_KEY`.

The redactor holds back only the tail of the text that could still become a match, and releases the rest straight away. That tail is the trailing run of characters that can be part of an email or a number. It ends at the first character outside those sets, so normal prose is held for about one word. Cards are checked with Luhn so order numbers and phone numbers pass through.

Measured with the mock provider on localhost (4-character chunks, 20 ms apart):

| | Direct | Through gateway |
| --- | --- | --- |
| First token | about 23 ms | about 46 ms |
| Whole stream | about 803 ms | about 806 ms |

The extra time to first token is one chunk interval, because the first word has to be held until the next chunk shows it is finished. Over 420,000 deltas, `python -m task3_pii_gateway.benchmark` showed 11.7 KiB peak memory, and memory is capped at one chunk plus about 570 characters however long the stream runs.

There is no queue between provider and client, so a slow client slows the provider read. A client disconnect closes the provider connection.

Things that could leak, and how they are handled:

- **Cut-off streams:** `john@gmail` matches no pattern, so a stream ended by `finish_reason: length` or a provider failure has its held tail redacted if it contains `@` or a digit.
- **Unparseable chunks:** dropped, because they can't be inspected.
- **Provider error bodies:** never forwarded, because they can echo the prompt.

Known limits:

- Only `delta.content` is scanned, not `tool_calls` arguments.
- Nine bare digits are not treated as an SSN, since too many ordinary numbers look like that.
- Regex can't catch PII that is spelled out, like "john at gmail dot com".
- A card inside an unbroken sequence of more than 48 digits can be missed.

## Task 4: Rate limiter and fallback router

```bash
GATEWAY_API_KEYS=tenant-a,tenant-b \
PRIMARY_URL=http://127.0.0.1:8101/v1/chat/completions PRIMARY_API_KEY=x PRIMARY_MODEL=model-a \
SECONDARY_URL=http://127.0.0.1:8102/v1/chat/completions SECONDARY_API_KEY=x SECONDARY_MODEL=model-b \
uvicorn task4_model_router.app:create_app --factory --port 8100
```

**Rate limiter.** Each admitted request is a row of tenant, timestamp and tokens, and usage is the sum of rows in the last 60 seconds. Tokens are reserved from an estimate before the provider call, then corrected to real usage afterwards. Otherwise a burst of concurrent requests would all pass while the counter still read zero. The check and insert run inside `BEGIN IMMEDIATE`, so the limit holds across worker processes, and a test runs four processes against one file to confirm it. Database work runs on one dedicated thread so the event loop never blocks on disk.

**Eviction.** A tenant's expired rows are deleted on its next request, and a periodic sweep clears tenants that have gone quiet. API keys are stored as SHA-256 hashes.

**Failover.** `asyncio.timeout` bounds the whole primary call, because httpx timeouts are per socket operation and a slow-dripping provider would never trip them. Timing out cancels the request and closes the connection, so a late primary answer can't arrive after failover. The task names 429 and timeouts. I also fail over on 5xx, connection errors and invalid JSON, since the backup can help with all of those. Other 4xx responses don't fail over, because the backup would refuse too.

**Token accounting.** A failover charges the tenant once. If all providers fail or the client disconnects, the reservation is released, and that release is shielded from cancellation.

**Errors.** Every failure uses one shape, and upstream detail stays in the server log under the request id:

```json
{"error": {"code": "service_unavailable", "message": "No model provider is available. Try again shortly.", "request_id": "req_..."}}
```

Next steps for production:

- Add a circuit breaker, because every request waits the full 3 seconds while the primary is down.
- Honor the provider's `Retry-After`.
- Use a real tokenizer for the estimate. The current one is 4 characters per token plus `max_tokens`.
- Move tenants and per-tenant limits into the database.
- SQLite suits one host. Several hosts would need a shared store such as Redis.
