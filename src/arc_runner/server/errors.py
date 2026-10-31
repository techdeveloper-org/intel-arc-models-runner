"""Client-safe error helpers: stable public messages out, full detail in the server log.

Raw exception text can hold absolute filesystem paths (and on Windows the user name), so routes must not copy
it into a response. These helpers log the detail with structured ``extra`` fields and return a fixed message.
"""

from typing import Any

from fastapi import HTTPException

from arc_runner.core.logging import get_logger

logger = get_logger("server.errors")

_MAX_NAME_CHARS = 128
_MAX_DETAIL_CHARS = 1000


def model_not_found_detail(model: str) -> str:
    """Build the client message for an unknown model from the requested name only.

    Args:
        model: Model name as requested by the client.

    Returns:
        str: "Model '<name>' was not found" with the name cut to a bounded length.
    """
    return f"Model '{str(model)[:_MAX_NAME_CHARS]}' was not found"


def hidden_error(log_message: str, exc: BaseException, public: str, status_code: int = 500, **context: Any) -> HTTPException:
    """Log an exception with its detail and return an HTTPException carrying only a stable message.

    Args:
        log_message: Fixed text for the log record.
        exc: The exception; its text and class go to the log, never to the client.
        public: Client-visible message.
        status_code: HTTP status of the response.
        **context: Extra key-value fields for the log record (for example the requested model).

    Returns:
        HTTPException: Ready to raise.
    """
    fields = {"error_class": type(exc).__name__, "error_detail": str(exc)[:_MAX_DETAIL_CHARS]}
    fields.update({f"ctx_{key}": str(value)[:_MAX_NAME_CHARS] for key, value in context.items()})
    logger.error(log_message, extra=fields, exc_info=exc)
    return HTTPException(status_code=status_code, detail=public)
