"""Browser-facing request guards: Host validation (DNS rebinding) and cross-site request checks (CSRF).

A loopback bind does not stop a web page in the user's browser from reaching the server: DNS rebinding makes the
page same-origin with it, and a cross-origin "simple" request (text/plain POST, an <img> GET) is sent even though
CORS then hides the response. These ASGI middlewares refuse such requests before any route runs.
"""

from functools import lru_cache
import ipaddress
import os
import re
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

from starlette.datastructures import Headers
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from arc_runner.core.logging import get_logger
from arc_runner.server.anthropic_schemas import error_envelope, new_request_id

logger = get_logger("server.security")

ALLOWED_HOSTS_ENV = "ARC_ALLOWED_HOSTS"
CORS_ORIGINS_ENV = "ARC_CORS_ORIGINS"
LOCAL_ORIGIN_REGEX = r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$"
ANTHROPIC_PATH_PREFIX = "/v1/messages"

NON_JSON_BODY_PATHS: Tuple[str, ...] = ()
"""Exact paths of mutating routes that legitimately take a non-JSON body (multipart or form). None exist today."""

STATE_CHANGING_GET_PATHS: Tuple[str, ...] = ("/api/gguf/convert/stream",)
"""GET routes that start work or write to disk; they get the same cross-site checks as POST."""

_MUTATING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
_HOST_NAME = re.compile(r"^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?(\.[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?)*$")
_PORT = re.compile(r"^[0-9]{1,5}$")
_LOCAL_ORIGIN = re.compile(LOCAL_ORIGIN_REGEX)


def _safe_repr(value: str, limit: int = 80) -> str:
    """Return a bounded, ASCII-escaped form of untrusted text for logging.

    Args:
        value: Untrusted header text.
        limit: Maximum characters kept.

    Returns:
        str: ascii() form cut to the limit.
    """
    return ascii(value)[:limit]


def _error_body(path: str, status: int, error_type: str, code: str, message: str) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Build an error body in the style of the route family the path belongs to.

    Args:
        path: Request path.
        status: HTTP status (unused for the body; kept for readability at call sites).
        error_type: Anthropic error type used on /v1/messages.
        code: Machine-readable code used elsewhere.
        message: Fixed message (never built from request data).

    Returns:
        Tuple[Dict[str, Any], Dict[str, str]]: JSON body and extra headers.
    """
    if path.startswith(ANTHROPIC_PATH_PREFIX):
        request_id = new_request_id()
        return error_envelope(error_type, message, request_id), {"request-id": request_id}
    return {"error": message, "code": code}, {}


async def _reject(scope: Scope, receive: Receive, send: Send, status: int, error_type: str, code: str, message: str) -> None:
    """Send a rejection (or close a websocket) and stop the request.

    Args:
        scope: ASGI scope.
        receive: ASGI receive.
        send: ASGI send.
        status: HTTP status.
        error_type: Anthropic error type for /v1/messages.
        code: Machine-readable code.
        message: Fixed client-visible message.
    """
    if scope["type"] == "websocket":
        await send({"type": "websocket.close", "code": 1008})
        return
    content, headers = _error_body(scope.get("path", ""), status, error_type, code, message)
    await JSONResponse(content, status_code=status, headers=headers)(scope, receive, send)


@lru_cache(maxsize=16)
def parse_allowed_hosts(raw: str) -> FrozenSet[str]:
    """Parse the ARC_ALLOWED_HOSTS value into a set of lower-case host names.

    Entries are comma separated bare host names. Whitespace around an entry is trimmed; entries that are not
    valid DNS names (ports, schemes, wildcards, slashes, inner spaces) are ignored with one warning.

    Args:
        raw: Raw environment value.

    Returns:
        FrozenSet[str]: Accepted names, lower case.
    """
    names: List[str] = []
    for entry in raw.split(","):
        candidate = entry.strip().lower()
        if not candidate:
            continue
        if len(candidate) <= 253 and _HOST_NAME.match(candidate):
            names.append(candidate)
        else:
            logger.warning("ignoring invalid ARC_ALLOWED_HOSTS entry", extra={"entry": _safe_repr(entry)})
    return frozenset(names)


def split_host_header(value: str) -> Optional[str]:
    """Extract the lower-case host from a Host header value, or None when it is malformed.

    Args:
        value: Raw Host header value.

    Returns:
        Optional[str]: Host name or IP literal without brackets and port; None for empty or malformed values.
    """
    text = value.strip()
    if not text or text != value or not text.isascii() or not text.isprintable() or " " in text:
        return None
    if text.startswith("["):
        end = text.find("]")
        if end < 0:
            return None
        host, rest = text[1:end], text[end + 1:]
        if rest and not (rest.startswith(":") and _PORT.match(rest[1:])):
            return None
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            return None
        return host.lower() if address.version == 6 and "%" not in host else None
    if text.count(":") > 1:
        return None
    host, sep, port = text.partition(":")
    if sep and not _PORT.match(port):
        return None
    host = host.lower()
    if host.endswith("."):
        host = host[:-1]
    return host or None


def host_allowed(value: str, extra_hosts: FrozenSet[str]) -> bool:
    """Decide whether a Host header value is acceptable.

    Args:
        value: Raw Host header value.
        extra_hosts: Names from ARC_ALLOWED_HOSTS.

    Returns:
        bool: True for localhost, any IP literal, or a name in extra_hosts (port ignored, case-insensitive).
    """
    host = split_host_header(value)
    if host is None:
        return False
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return host == "localhost" or host in extra_hosts


class HostValidationMiddleware:
    """Refuses requests whose Host header is not localhost, an IP literal or listed in ARC_ALLOWED_HOSTS (421)."""

    def __init__(self, app: ASGIApp) -> None:
        """Store the downstream application.

        Args:
            app: Downstream ASGI application.
        """
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Validate the Host header, then continue or answer 421.

        Args:
            scope: ASGI scope.
            receive: ASGI receive.
            send: ASGI send.
        """
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        values = [value for key, value in scope["headers"] if key == b"host"]
        extra = parse_allowed_hosts(os.environ.get(ALLOWED_HOSTS_ENV, ""))
        if len(values) == 1 and host_allowed(values[0].decode("latin-1"), extra):
            await self.app(scope, receive, send)
            return
        shown = _safe_repr(", ".join(v.decode("latin-1") for v in values))
        logger.warning("rejected request with a disallowed Host header", extra={"bad_host": shown, "host_count": len(values)})
        await _reject(scope, receive, send, 421, "invalid_request_error", "HOST_NOT_ALLOWED", "Host header not allowed")


class BrowserGuardMiddleware:
    """Refuses cross-site browser requests: foreign Origin (403) and non-JSON bodies on mutating methods (415)."""

    def __init__(self, app: ASGIApp, extra_origins: Sequence[str] = ()) -> None:
        """Store the downstream application and the extra allowed origins.

        Args:
            app: Downstream ASGI application.
            extra_origins: Origins from ARC_CORS_ORIGINS ("*" allows every origin, as it does for CORS).
        """
        self.app = app
        self._extra = frozenset(extra_origins)

    def _origin_allowed(self, origin: str, host: Optional[str]) -> bool:
        """Apply the local-origin policy.

        Args:
            origin: Origin header value.
            host: Validated Host header value (same-origin requests carry it as their own origin).

        Returns:
            bool: True for the local-origin regex, ARC_CORS_ORIGINS entries, or the server's own origin.
        """
        if "*" in self._extra or origin in self._extra or _LOCAL_ORIGIN.fullmatch(origin):
            return True
        return bool(host) and origin.lower() in (f"http://{host.lower()}", f"https://{host.lower()}")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Check Origin and Content-Type, then continue or reject.

        Args:
            scope: ASGI scope.
            receive: ASGI receive.
            send: ASGI send.
        """
        if scope["type"] != "http" or scope["method"] == "OPTIONS":
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        method, path = scope["method"], scope["path"]
        origin = headers.get("origin")
        state_changing = method in _MUTATING_METHODS or path in STATE_CHANGING_GET_PATHS
        if origin is not None and not self._origin_allowed(origin, headers.get("host")):
            logger.warning("rejected cross-origin request", extra={"bad_origin": _safe_repr(origin), "method": method})
            await _reject(scope, receive, send, 403, "permission_error", "ORIGIN_NOT_ALLOWED", "Cross-origin request not allowed")
            return
        if origin is None and state_changing and headers.get("sec-fetch-site", "").lower() == "cross-site":
            logger.warning("rejected cross-site request without Origin", extra={"method": method})
            await _reject(scope, receive, send, 403, "permission_error", "ORIGIN_NOT_ALLOWED", "Cross-origin request not allowed")
            return
        if method in _MUTATING_METHODS and path not in NON_JSON_BODY_PATHS and _has_body(headers):
            media_type = headers.get("content-type", "").split(";", 1)[0].strip().lower()
            if media_type != "application/json":
                await _reject(scope, receive, send, 415, "invalid_request_error", "UNSUPPORTED_MEDIA_TYPE", "Content-Type must be application/json")
                return
        await self.app(scope, receive, send)


def _has_body(headers: Headers) -> bool:
    """Tell whether the request declares a body.

    Args:
        headers: Request headers.

    Returns:
        bool: True for a positive Content-Length or any Transfer-Encoding.
    """
    length = headers.get("content-length", "")
    return (length.isdigit() and int(length) > 0) or "transfer-encoding" in headers
