"""JSON-RPC parsing and the tool authorization rule."""

import json
from typing import Any

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
UNAUTHENTICATED = -32000
UNAUTHORIZED_TOOL_CALL = -32001

ADMIN_TOOL_PREFIX = "admin_"


def error_response(request_id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


class DuplicateKey(ValueError):
    pass


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateKey(key)
        result[key] = value
    return result


def parse(body: bytes) -> Any:
    """Parses a payload. Raises ValueError for anything that isn't clean JSON.

    Duplicate keys are refused because parsers disagree on them. With
    {"method": "tools/call", "method": "tools/list"} this gateway reads the last
    value and a downstream server might read the first, so a call could get past
    authorization looking like a list.
    """
    return json.loads(body, object_pairs_hook=_reject_duplicate_keys)


def check_message(message: Any, role: str) -> dict | None:
    """Returns an error response if this message must not be forwarded, otherwise None."""
    if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
        return error_response(None, INVALID_REQUEST, "Invalid Request")

    request_id = message.get("id")
    method = message.get("method")
    if method is None:
        return None  # a client's reply to a server-initiated request
    if not isinstance(method, str):
        return error_response(request_id, INVALID_REQUEST, "Invalid Request")
    if method != "tools/call":
        return None

    params = message.get("params")
    name = params.get("name") if isinstance(params, dict) else None
    if not isinstance(name, str) or not name:
        return error_response(request_id, INVALID_PARAMS, "params.name must be a non-empty string")

    # Normalized so " Admin_reset" can't slip past and then match on a lenient
    # downstream server. The role check is exact: an unknown role is not admin.
    if name.strip().lower().startswith(ADMIN_TOOL_PREFIX) and role != "admin":
        return error_response(request_id, UNAUTHORIZED_TOOL_CALL, "Unauthorized Tool Call")
    return None
