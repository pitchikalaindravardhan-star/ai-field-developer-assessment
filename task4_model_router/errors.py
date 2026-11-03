"""The only error shape clients ever see from the gateway."""

from starlette.responses import JSONResponse


class GatewayError(Exception):
    def __init__(self, status: int, code: str, message: str, retry_after: int | None = None):
        super().__init__(code)
        self.status = status
        self.code = code
        self.message = message
        self.retry_after = retry_after


def invalid_request(message: str) -> GatewayError:
    return GatewayError(400, "invalid_request", message)


def unauthorized() -> GatewayError:
    return GatewayError(401, "unauthorized", "Missing or invalid API key.")


def rate_limited(retry_after: int) -> GatewayError:
    return GatewayError(429, "rate_limit_exceeded", "Token rate limit exceeded.", retry_after)


def request_too_large() -> GatewayError:
    return GatewayError(413, "request_too_large", "Request exceeds the tenant's token limit.")


def upstream_rejected() -> GatewayError:
    return GatewayError(502, "upstream_error", "The request could not be completed.")


def unavailable() -> GatewayError:
    return GatewayError(503, "service_unavailable", "No model provider is available. Try again shortly.")


def gateway_timeout() -> GatewayError:
    return GatewayError(504, "gateway_timeout", "The model providers did not respond in time.")


def internal_error() -> GatewayError:
    return GatewayError(500, "internal_error", "Something went wrong on our side.")


def error_response(error: GatewayError, request_id: str) -> JSONResponse:
    headers = {"X-Request-Id": request_id}
    if error.retry_after is not None:
        headers["Retry-After"] = str(error.retry_after)
    return JSONResponse(
        {"error": {"code": error.code, "message": error.message, "request_id": request_id}},
        status_code=error.status,
        headers=headers,
    )
