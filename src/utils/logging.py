"""Logging configuration for F1 Race Insight."""

import logging
import sys
from typing import Any, Dict, Optional

import coloredlogs
import structlog

# Set up standard Python logging
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)

# Configure structlog
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

# Configure coloredlogs
coloredlogs.install(
    level="INFO",
    fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger with the given name.

    Args:
        name: The name of the logger.

    Returns:
        A configured structlog logger.
    """
    return structlog.get_logger(name)


def log_function_call(
    logger: structlog.stdlib.BoundLogger,
    func_name: str,
    args: Optional[Dict[str, Any]] = None,
) -> None:
    """Log a function call with arguments.

    Args:
        logger: The structlog logger to use.
        func_name: The name of the function being called.
        args: The arguments being passed to the function.
    """
    logger.info(f"Calling {func_name}", args=args if args else {}) 