"""Centralized structured logging subsystem with MDC request tracing for Intel Arc Models Runner.

Provides dual-appender logging (Rich console + Rotating JSON file) compliant with
Rule 27 standards, incorporating ContextVar request tracing (X-Request-ID) and non-blocking
log emission.
"""

import contextvars
import json
import logging
import logging.handlers
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from rich.logging import RichHandler

_current_request_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("request_id", default=None)
_current_client_ip: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("client_ip", default=None)
_current_model_name: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("model_name", default=None)
_current_target_device: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("target_device", default=None)


def set_request_context(
    request_id: Optional[str] = None,
    client_ip: Optional[str] = None,
    model_name: Optional[str] = None,
    target_device: Optional[str] = None,
) -> None:
    """Set the active MDC request tracing context in current async task context.

    Args:
        request_id: Unique request identifier or correlation token.
        client_ip: Remote client IP address string.
        model_name: Active model identifier being queried.
        target_device: Hardware silicon target (GPU, NPU, CPU).
    """
    if request_id is not None:
        _current_request_id.set(request_id)
    if client_ip is not None:
        _current_client_ip.set(client_ip)
    if model_name is not None:
        _current_model_name.set(model_name)
    if target_device is not None:
        _current_target_device.set(target_device)


def clear_request_context() -> None:
    """Clear all MDC request tracing context variables."""
    _current_request_id.set(None)
    _current_client_ip.set(None)
    _current_model_name.set(None)
    _current_target_device.set(None)


def get_request_id() -> Optional[str]:
    """Retrieve current active request correlation ID.

    Returns:
        The current request ID string, or None if no context is active.
    """
    return _current_request_id.get()


class RequestContextFilter(logging.Filter):
    """Logging filter that injects active MDC context attributes into LogRecord instances."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Enrich log record with active correlation ID and device context.

        Args:
            record: Standard logging record instance.

        Returns:
            Always True to allow the record to proceed through the handler chain.
        """
        record.request_id = _current_request_id.get() or "system"
        record.client_ip = _current_client_ip.get() or "-"
        record.model_name = _current_model_name.get() or "-"
        record.target_device = _current_target_device.get() or "-"
        return True


class JSONFileFormatter(logging.Formatter):
    """JSON structured log formatter for file appenders complying with Rule 27."""

    def format(self, record: logging.LogRecord) -> str:
        """Format the specified log record as a structured single-line JSON string.

        Args:
            record: Standard logging record instance.

        Returns:
            Single-line JSON formatted string with standard metadata and metrics.
        """
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "service": "intel-arc-models-runner",
            "logger": record.name,
            "request_id": getattr(record, "request_id", "system"),
            "client_ip": getattr(record, "client_ip", "-"),
            "model_name": getattr(record, "model_name", "-"),
            "target_device": getattr(record, "target_device", "-"),
            "message": record.getMessage(),
        }

        if hasattr(record, "metrics") and isinstance(record.metrics, dict):
            log_entry["metrics"] = record.metrics

        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    debug: bool = False,
) -> None:
    """Configure centralized dual-appender logging with MDC filter.

    Args:
        level: Minimum logging level name (e.g. 'DEBUG', 'INFO', 'WARNING').
        log_file: Optional path to rotating JSON file on disk.
        debug: If True, overrides level to DEBUG and enables verbose formatting.
    """
    resolved_level = logging.DEBUG if debug else getattr(logging, level.upper(), logging.INFO)
    logger = logging.getLogger("arc_runner")
    logger.setLevel(resolved_level)
    logger.handlers.clear()

    mdc_filter = RequestContextFilter()

    console_handler = RichHandler(
        level=resolved_level,
        show_time=True,
        show_level=True,
        show_path=debug,
        rich_tracebacks=True,
        markup=True,
    )
    console_handler.setLevel(resolved_level)
    console_handler.addFilter(mdc_filter)
    console_format = logging.Formatter("[req:%(request_id)s] %(message)s")
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    target_file = log_file or Path("arc-runner.log")
    file_handler = logging.handlers.RotatingFileHandler(
        filename=str(target_file),
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(resolved_level)
    file_handler.addFilter(mdc_filter)
    file_handler.setFormatter(JSONFileFormatter())
    logger.addHandler(file_handler)

    logger.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Retrieve an initialized logger scoped under the arc_runner namespace.

    Args:
        name: Subsystem or module name.

    Returns:
        Configured Logger instance.
    """
    if name.startswith("arc_runner"):
        return logging.getLogger(name)
    return logging.getLogger(f"arc_runner.{name}")
