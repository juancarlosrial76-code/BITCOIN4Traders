"""
Unified Logging Configuration
==============================
Routes all standard-library ``logging.getLogger()`` calls through loguru,
so modules that use ``import logging`` and modules that use
``from loguru import logger`` produce a single, consistent log stream.

Usage — call once at application startup (e.g. in run.py or main.py):

    from src.config.logging_config import setup_logging
    setup_logging(level="INFO", log_file="logs/trading.log")

After this call every ``logging.getLogger("...")`` record is handled by
loguru.  No changes to existing modules are required.
"""

import logging
import sys
from typing import Optional

from loguru import logger


class _InterceptHandler(logging.Handler):
    """
    Intercept handler that redirects standard-library ``logging`` records
    to loguru.  This bridges the two logging systems without modifying
    individual modules.
    """

    def emit(self, record: logging.LogRecord) -> None:
        # Map standard level to loguru level name
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = str(record.levelno)

        # Find correct caller depth so loguru reports the *original* call site
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back  # type: ignore[assignment]
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    rotation: str = "10 MB",
    retention: str = "7 days",
) -> None:
    """
    Configure unified logging for the BITCOIN4Traders application.

    Parameters
    ----------
    level : str
        Minimum log level (DEBUG / INFO / WARNING / ERROR / CRITICAL).
    log_file : str, optional
        Path to a rotating log file.  If None, logs go to stderr only.
    rotation : str
        loguru rotation policy (e.g. "10 MB", "1 day").
    retention : str
        loguru retention policy (e.g. "7 days").
    """
    # Remove default loguru handler and add a clean one
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    if log_file:
        logger.add(
            log_file,
            level=level,
            rotation=rotation,
            retention=retention,
            compression="zip",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{line} - {message}",
        )

    # Intercept all standard-library logging records
    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)

    # Silence noisy third-party loggers
    for noisy in ("urllib3", "aiohttp", "asyncio", "websockets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logger.info(
        f"Logging configured: level={level}"
        + (f" file={log_file}" if log_file else "")
    )
