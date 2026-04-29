"""
InsightSerenity AI Engine — Structured Logger
=============================================
Provides a consistent, structured logging interface used across every module
in the ai-engine. In development it emits human-readable coloured output;
in production it emits newline-delimited JSON for ingestion by log aggregators
(Loki, Elasticsearch, CloudWatch, etc.).

Usage:
    from src.utils.logger import get_logger

    logger = get_logger(__name__)
    logger.info("Training started", epoch=1, lr=3e-4)
    logger.error("Checkpoint save failed", path=str(path), exc_info=True)
"""

import logging
import sys
import time
import json
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from src.config.settings import settings, Environment, LogLevel

_LOG_RECORD_RESERVED_KEYS = {
    "name", "msg", "args", "levelname", "levelno", "pathname",
    "filename", "module", "exc_info", "exc_text", "stack_info",
    "lineno", "funcName", "created", "msecs", "relativeCreated",
    "thread", "threadName", "processName", "process", "message",
    "taskName",
}


# ─────────────────────────────────────────────────────────────────────────────
# ANSI colour codes (used only in development pretty-printing)
# ─────────────────────────────────────────────────────────────────────────────

_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_GREY   = "\033[90m"
_CYAN   = "\033[96m"
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_MAGENTA = "\033[95m"

_LEVEL_COLOURS: Dict[int, str] = {
    logging.DEBUG:    _GREY,
    logging.INFO:     _GREEN,
    logging.WARNING:  _YELLOW,
    logging.ERROR:    _RED,
    logging.CRITICAL: _MAGENTA,
}


# ─────────────────────────────────────────────────────────────────────────────
# Pretty formatter (development)
# ─────────────────────────────────────────────────────────────────────────────

class PrettyFormatter(logging.Formatter):
    """
    Human-readable log formatter for local development.
    Outputs:  HH:MM:SS  LEVEL  module_name  message  {extra_fields}

    Example:
        14:32:01  INFO  src.data.crawler  Crawl started  {"seed_count": 5}
    """

    def format(self, record: logging.LogRecord) -> str:
        # Timestamp
        ts = datetime.fromtimestamp(record.created, tz=timezone.utc).strftime(
            "%H:%M:%S"
        )

        # Level with colour
        colour     = _LEVEL_COLOURS.get(record.levelno, _RESET)
        level_str  = f"{colour}{_BOLD}{record.levelname:<8}{_RESET}"

        # Module name (shortened)
        module     = f"{_CYAN}{record.name}{_RESET}"

        # Main message
        message    = record.getMessage()

        # Extra structured fields (everything not standard)
        extra = {
            k: v
            for k, v in record.__dict__.items()
            if k not in _LOG_RECORD_RESERVED_KEYS
        }
        extra_str = f"  {_GREY}{json.dumps(extra)}{_RESET}" if extra else ""

        # Exception info
        exc_str = ""
        if record.exc_info:
            exc_str = "\n" + self.formatException(record.exc_info)

        return (
            f"{_GREY}{ts}{_RESET}  {level_str}  {module}  {message}"
            f"{extra_str}{exc_str}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# JSON formatter (staging / production)
# ─────────────────────────────────────────────────────────────────────────────

class JSONFormatter(logging.Formatter):
    """
    Newline-delimited JSON log formatter for production environments.
    Every log line is a complete, self-contained JSON object compatible with
    Loki, Elasticsearch, CloudWatch Logs, and most log aggregation systems.

    Standard fields always present:
        timestamp, level, logger, message, environment, version
    """

    def format(self, record: logging.LogRecord) -> str:
        # Base log record
        log_entry: Dict[str, Any] = {
            "timestamp":   datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level":       record.levelname,
            "logger":      record.name,
            "message":     record.getMessage(),
            "environment": settings.environment.value,
            "version":     settings.version,
            "module":      record.module,
            "function":    record.funcName,
            "line":        record.lineno,
        }

        # Merge any extra keyword arguments passed to the logger call
        for key, value in record.__dict__.items():
            if key not in _LOG_RECORD_RESERVED_KEYS:
                log_entry[key] = value

        # Exception details
        if record.exc_info:
            log_entry["exception"] = {
                "type":    record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }

        return json.dumps(log_entry, default=str)


# ─────────────────────────────────────────────────────────────────────────────
# ContextLogger — adds structured keyword arguments to every call
# ─────────────────────────────────────────────────────────────────────────────

class ContextLogger(logging.Logger):
    """
    Extended Logger that accepts keyword arguments and attaches them as
    structured fields to the log record. This enables zero-cost structured
    logging without string formatting.

    Example:
        logger.info("Batch processed", batch_idx=42, loss=0.341)
        # → {"message": "Batch processed", "batch_idx": 42, "loss": 0.341, ...}
    """

    def _log_with_extras(
        self,
        level: int,
        msg: str,
        args: tuple,
        exc_info=None,
        extra: Optional[Dict[str, Any]] = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        **kwargs: Any,
    ) -> None:
        """Internal helper that merges kwargs into the `extra` dict."""
        merged_extra = {
            (f"context_{key}" if key in _LOG_RECORD_RESERVED_KEYS else key): value
            for key, value in {**(extra or {}), **kwargs}.items()
        }
        super()._log(
            level, msg, args,
            exc_info=exc_info,
            extra=merged_extra,
            stack_info=stack_info,
            stacklevel=stacklevel + 1,
        )

    def debug(self, msg: str, *args, **kwargs) -> None:       # type: ignore[override]
        self._log_with_extras(logging.DEBUG,    msg, args, **kwargs)

    def info(self, msg: str, *args, **kwargs) -> None:        # type: ignore[override]
        self._log_with_extras(logging.INFO,     msg, args, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:     # type: ignore[override]
        self._log_with_extras(logging.WARNING,  msg, args, **kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:       # type: ignore[override]
        self._log_with_extras(logging.ERROR,    msg, args, **kwargs)

    def critical(self, msg: str, *args, **kwargs) -> None:    # type: ignore[override]
        self._log_with_extras(logging.CRITICAL, msg, args, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Factory and root configuration
# ─────────────────────────────────────────────────────────────────────────────

# Register our custom logger class so all child loggers use ContextLogger
logging.setLoggerClass(ContextLogger)

# Resolve numeric log level
_LEVEL_MAP = {
    LogLevel.DEBUG:    logging.DEBUG,
    LogLevel.INFO:     logging.INFO,
    LogLevel.WARNING:  logging.WARNING,
    LogLevel.ERROR:    logging.ERROR,
    LogLevel.CRITICAL: logging.CRITICAL,
}
_ROOT_LEVEL = _LEVEL_MAP.get(settings.log_level, logging.INFO)


def _build_handler() -> logging.StreamHandler:
    """Create and configure the appropriate stream handler for the environment."""
    handler = logging.StreamHandler(stream=sys.__stdout__)
    handler.setLevel(_ROOT_LEVEL)

    if settings.is_development:
        handler.setFormatter(PrettyFormatter())
    else:
        handler.setFormatter(JSONFormatter())

    return handler


def _configure_root_logger() -> None:
    """
    Configure the root logger once at module import time.
    Subsequent calls to get_logger() inherit this configuration.
    """
    root = logging.getLogger()
    root.setLevel(_ROOT_LEVEL)
    logging.raiseExceptions = settings.debug

    # Avoid duplicate handlers if this module is re-imported in tests
    if not root.handlers:
        root.addHandler(_build_handler())

    # Silence overly verbose third-party loggers in non-debug mode
    if not settings.debug:
        for noisy in ("httpx", "httpcore", "asyncio", "urllib3", "charset_normalizer"):
            logging.getLogger(noisy).setLevel(logging.WARNING)


_configure_root_logger()


def get_logger(name: str) -> ContextLogger:
    """
    Return a named ContextLogger for the given module.

    Always call this at module level:
        logger = get_logger(__name__)

    Args:
        name: Typically __name__ of the calling module.

    Returns:
        A ContextLogger that supports structured keyword arguments.
    """
    logger = logging.getLogger(name)
    return logger  # type: ignore[return-value]  # setLoggerClass ensures correct type


# ─────────────────────────────────────────────────────────────────────────────
# Timer context manager — log operation durations with one line
# ─────────────────────────────────────────────────────────────────────────────

class LogTimer:
    """
    Context manager that logs the wall-clock duration of a block.

    Usage:
        with LogTimer(logger, "Tokenizer training", vocab_size=32000):
            tokenizer.train(corpus)
        # → INFO  Tokenizer training completed  {"duration_s": 12.34, "vocab_size": 32000}
    """

    def __init__(
        self,
        logger: ContextLogger,
        operation: str,
        level: int = logging.INFO,
        **context: Any,
    ) -> None:
        self._logger    = logger
        self._operation = operation
        self._level     = level
        self._context   = context
        self._start: float = 0.0

    def __enter__(self) -> "LogTimer":
        self._start = time.perf_counter()
        self._logger._log_with_extras(
            self._level,
            f"{self._operation} started",
            (),
            **self._context,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        duration = time.perf_counter() - self._start

        if exc_type is not None:
            self._logger.error(
                f"{self._operation} failed",
                duration_s=round(duration, 4),
                exc_info=(exc_type, exc_val, exc_tb),
                **self._context,
            )
        else:
            self._logger._log_with_extras(
                self._level,
                f"{self._operation} completed",
                (),
                duration_s=round(duration, 4),
                **self._context,
            )
        return False   # Never suppress exceptions
