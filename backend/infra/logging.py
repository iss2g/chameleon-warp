"""
Structured logging with per-request correlation.

What this gives you:

  • A `request_id` is assigned to every incoming request (from the
    `X-Request-Id` header if the upstream proxy already minted one,
    otherwise a fresh UUID).
  • A contextvar carries the id so any `log.info(...)` inside the
    handler is tagged without manual plumbing.
  • Logs are line-oriented JSON when `DT_LOG_JSON=1`, or a compact
    `[timestamp] [level] [req_id] message` text format otherwise (better
    for `journalctl -f`).

Wire-up: call `setup_logging()` once at process start (from main.py).
Then use `logging.getLogger(__name__)` everywhere. The `RequestIdMiddleware`
is what populates the contextvar.

This is intentionally tiny — no log shipping, no rotation, no async
queue. Production log aggregation (loki / datadog / cloudwatch) reads
from stdout, so we just write a clean structured stream there and let
the platform handle the rest.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from contextvars import ContextVar
from typing import Optional

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

# Context variable: empty when no request is in flight (e.g. during
# startup or in the TTL sweeper thread).
_request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)


def current_request_id() -> Optional[str]:
    """Return the request id of the currently-handled request, or None."""
    return _request_id_var.get()


# ----------------------------------------------------------------------- #
# Formatters                                                              #
# ----------------------------------------------------------------------- #

class _TextFormatter(logging.Formatter):
    """Compact human-readable line format with request id tag."""

    def format(self, record: logging.LogRecord) -> str:
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(record.created))
        rid = current_request_id() or "-"
        line = f"{ts}Z [{record.levelname}] [{rid}] {record.name}: {record.getMessage()}"
        # Always append the traceback for exceptions — otherwise 500s log only
        # "Exception in ASGI application" with no stack, which is useless.
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)
        return line


class _JsonFormatter(logging.Formatter):
    """Single-line JSON per record — easy to ship to log aggregators."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": current_request_id(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(level: str = "INFO") -> None:
    """
    Replace the root handlers so everything routed through `logging`
    follows our format.

    `print(...)` statements scattered around dt_core and pipeline timings
    still go straight to stdout untagged — that's fine, they're meant
    for ops observability and not user-correlated.
    """
    root = logging.getLogger()
    root.setLevel(level)
    # Remove any default uvicorn / FastAPI handlers so we don't double-emit.
    for h in list(root.handlers):
        root.removeHandler(h)
    formatter: logging.Formatter = (
        _JsonFormatter() if os.environ.get("DT_LOG_JSON", "0") == "1" else _TextFormatter()
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root.addHandler(handler)

    # Optional size-based rotating file handler for self-contained deployments
    # (no external log shipper). Enable by setting DT_LOG_FILE; tune size and
    # backup count with DT_LOG_MAX_BYTES / DT_LOG_BACKUPS.
    log_file = os.environ.get("DT_LOG_FILE")
    if log_file:
        from logging.handlers import RotatingFileHandler

        try:
            max_bytes = int(os.environ.get("DT_LOG_MAX_BYTES", str(20 * 1024 * 1024)))
        except ValueError:
            max_bytes = 20 * 1024 * 1024
        try:
            backups = int(os.environ.get("DT_LOG_BACKUPS", "5"))
        except ValueError:
            backups = 5
        try:
            from pathlib import Path

            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            fh = RotatingFileHandler(
                log_file, maxBytes=max_bytes, backupCount=backups, encoding="utf-8"
            )
            fh.setFormatter(formatter)
            root.addHandler(fh)
        except OSError as e:
            root.warning("could not open DT_LOG_FILE=%s for rotation: %s", log_file, e)

    # Uvicorn has its own loggers that publish access lines. Quiet them
    # by routing through our formatter at INFO+ (default is WARN, we
    # leave that alone so its existing INFO access log stays visible).
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers = []
        lg.propagate = True


# ----------------------------------------------------------------------- #
# Middleware                                                              #
# ----------------------------------------------------------------------- #

class RequestIdMiddleware(BaseHTTPMiddleware):
    """
    Assign / propagate `X-Request-Id` per request and stash it in the
    contextvar so loggers can include it.

    If the upstream proxy already added an X-Request-Id (some load
    balancers do) we trust it — that lets you trace a request across
    your edge logs and our app logs by a single id.
    """

    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        token = _request_id_var.set(rid)
        try:
            response = await call_next(request)
        finally:
            _request_id_var.reset(token)
        response.headers["X-Request-Id"] = rid
        return response
