"""
Logging configuration utilities for CantioAI.
"""

import logging
import sys
from typing import Optional
from pathlib import Path

# Logging format
DEFAULT_FORMAT = (
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    level: Union[int, str] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    format_str: Optional[str] = None,
    date_format: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        level: Logging level (e.g., logging.INFO, "DEBUG")
        log_file: Path to log file (None for stdout only)
        format_str: Log format string
        date_format: Date format string

    Returns:
        Configured root logger
    """
    # Convert level to int if string
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    # Set format
    if format_str is None:
        format_str = DEFAULT_FORMAT
    if date_format is None:
        date_format = DATE_FORMAT

    formatter = logging.Formatter(fmt=format_str, datefmt=date_format)

    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Add file handler if specified
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger (avoid duplicate logs)
    logger.propagate = False

    logger.info(f"Logging configured - level: {level}, file: {log_file}")

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get or create a logger with the specified name.

    Args:
        name: Logger name (None for root logger)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class LoggerAdapter:
    """
    Adapter to add contextual information to loggers.
    """

    def __init__(self, logger: logging.Logger, **context):
        """
        Initialize logger adapter.

        Args:
            logger: Logger to adapt
            **context: Contextual information to add to log records
        """
        self.logger = logger
        self.context = context

    def debug(self, msg: str, *args) -> None:
        """Log a message with level DEBUG."""
        if self.logger.isEnabledFor(logging.DEBUG):
            self._log(logging.DEBUG, msg, args)

    def info(self, msg: str, *args) -> None:
        """Log a message with level INFO."""
        if self.logger.isEnabledFor(logging.INFO):
            self._log(logging.INFO, msg, args)

    def warning(self, msg: str, *args) -> None:
        """Log a message with level WARNING."""
        if self.logger.isEnabledFor(logging.WARNING):
            self._log(logging.WARNING, msg, args)

    def error(self, msg: str, *args) -> None:
        """Log a message with level ERROR."""
        if self.logger.isEnabledFor(logging.ERROR):
            self._log(logging.ERROR, msg, args)

    def critical(self, msg: str, *args) -> None:
        """Log a message with level CRITICAL."""
        if self.logger.isEnabledFor(logging.CRITICAL):
            self._log(logging.CRITICAL, msg, args)

    def _log(self, level: int, msg: str, args: tuple) -> None:
        """Internal logging method."""
        if self.logger.isEnabledFor(level):
            # Format message with context
            full_msg = msg
            if args:
                full_msg = msg % args
            # Add context as extra
            extra = {
                **self.context
            }
            # Call logger with extra
            self.logger.handle(self.logger.makeRecord(
                level, self.logger.getName(), full_msg, extra, None, None
            ))


if __name__ == "__main__":
    # Simple test
    import tempfile
    import os

    # Test basic setup
    logger = setup_logging(logging.DEBUG)
    logger.info("Test INFO message")
    logger.debug("Test DEBUG message")
    logger.warning("Test WARNING message")
    logger.error("Test ERROR message")

    # Test file logging
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
        log_file = f.name
        logger = setup_logging(logging.INFO, log_file=log_file)
        logger.info("Test file INFO message")
        logger.debug("Test file DEBUG message")

        # Verify file was written
        with open(log_file, 'r') as f:
            content = f.read()
            assert "Test file INFO message" in content
            assert "Test file DEBUG message" in content

        # Clean up
        os.unlink(log_file)

    # Test logger adapter
    logger = setup_logging()
    adapter = LoggerAdapter(logger, user="test", session="123")
    adapter.info("Test adapter message")
    adapter.warning("Test adapter warning")

    print("Logging utilities test passed.")