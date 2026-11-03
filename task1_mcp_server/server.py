"""MCP server exposing get_customer_record and trigger_refund over stdio.

Run with:  python -m task1_mcp_server.server

Error handling follows the split in the MCP spec:
  * the request itself is wrong (unknown tool, bad arguments) -> JSON-RPC error
    with code -32602, so the client sees a protocol failure.
  * the request was fine but the business operation failed (no such customer,
    balance too low) -> a normal result with isError=true, which the model can
    read and react to.

The low-level Server is used instead of MCPServer because MCPServer reports
argument validation failures as isError results, not as JSON-RPC errors.
"""

import json
import logging
import sys
from typing import Any

import anyio
import mcp_types as types
from mcp.server import Server, ServerRequestContext
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import MCPError
from pydantic import BaseModel, ValidationError

from .schemas import GetCustomerRecordInput, TriggerRefundInput
from .store import CustomerNotFound, CustomerStore, RefundExceedsBalance

log = logging.getLogger("customer-mcp")
store = CustomerStore()


def _text(payload: dict[str, Any], is_error: bool = False) -> types.CallToolResult:
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=json.dumps(payload))],
        is_error=is_error,
    )


def _get_customer_record(args: GetCustomerRecordInput) -> types.CallToolResult:
    return _text(store.get(args.customer_id).public_view())


def _trigger_refund(args: TriggerRefundInput) -> types.CallToolResult:
    return _text(store.refund(args.customer_id, args.amount, args.reason))


TOOLS: dict[str, tuple[str, type[BaseModel], Any, types.ToolAnnotations]] = {
    "get_customer_record": (
        "Look up a customer record by id.",
        GetCustomerRecordInput,
        _get_customer_record,
        types.ToolAnnotations(read_only_hint=True),
    ),
    "trigger_refund": (
        "Issue a refund against a customer's refundable balance.",
        TriggerRefundInput,
        _trigger_refund,
        types.ToolAnnotations(destructive_hint=True, idempotent_hint=False),
    ),
}


async def list_tools(
    ctx: ServerRequestContext, params: types.PaginatedRequestParams | None
) -> types.ListToolsResult:
    return types.ListToolsResult(
        tools=[
            types.Tool(
                name=name,
                description=description,
                input_schema=model.model_json_schema(),
                annotations=annotations,
            )
            for name, (description, model, _, annotations) in TOOLS.items()
        ]
    )


def _validation_error(tool: str, exc: ValidationError) -> MCPError:
    # Field, reason and rule only. The rejected value stays out of the response
    # and the logs because it may be sensitive.
    details = [
        {
            "field": ".".join(str(part) for part in err["loc"]) or "arguments",
            "type": err["type"],
            "message": err["msg"],
        }
        for err in exc.errors(include_url=False, include_input=False, include_context=False)
    ]
    log.info("rejected %s call: %s", tool, [d["field"] for d in details])
    return MCPError(types.INVALID_PARAMS, f"Invalid arguments for {tool}", {"errors": details})


async def call_tool(
    ctx: ServerRequestContext, params: types.CallToolRequestParams
) -> types.CallToolResult:
    if params.name not in TOOLS:
        raise MCPError(types.INVALID_PARAMS, f"Unknown tool: {params.name}")

    _, model, handler, _ = TOOLS[params.name]
    try:
        args = model.model_validate(params.arguments or {})
    except ValidationError as exc:
        raise _validation_error(params.name, exc) from exc

    try:
        result = handler(args)
    except CustomerNotFound:
        return _text({"error": "customer_not_found"}, is_error=True)
    except RefundExceedsBalance:
        return _text({"error": "refund_exceeds_balance"}, is_error=True)
    except Exception:
        log.exception("unhandled failure in %s", params.name)
        raise MCPError(types.INTERNAL_ERROR, "Internal server error") from None

    log.info("tool call completed: %s", params.name)
    return result


server = Server(
    "customer-support",
    version="1.0.0",
    on_list_tools=list_tools,
    on_call_tool=call_tool,
)


def configure_logging() -> None:
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=True,
    )


async def run() -> None:
    log.info("starting on stdio")
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main() -> None:
    configure_logging()
    # stdio_server() moves the real stdout to a private descriptor and points
    # fd 1 at stderr while it runs, so a stray print() or a child process can't
    # corrupt the stream. Don't swap sys.stdout here: the transport needs to find
    # it on fd 1 to do that. test_stdio.py covers both cases.
    anyio.run(run)


if __name__ == "__main__":
    main()
