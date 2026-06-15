"""Logging utilities."""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


class _ColorFormatter(logging.Formatter):
    """Colored formatter for console output only."""

    COLORS = {
        logging.DEBUG: "\033[36m",
        logging.INFO: "\033[32m",
        logging.WARNING: "\033[33m",
        logging.ERROR: "\033[31m",
        logging.CRITICAL: "\033[41m",
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        color = self.COLORS.get(record.levelno, self.RESET)
        return f"{color}{message}{self.RESET}"


class Logger:
    """Simple experiment logger wrapping Python logging."""

    def __init__(self, log_dir: str | Path, name: str = "train") -> None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{name}_{timestamp}.log"

        self.logger = logging.getLogger(f"{name}_{timestamp}")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        self.logger.propagate = False

        plain_formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        color_formatter = _ColorFormatter(
            "[%(asctime)s] %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(plain_formatter)
        self.logger.addHandler(fh)

        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(color_formatter)
        self.logger.addHandler(sh)

    def info(self, msg: str) -> None:
        self.logger.info(msg)

    def warning(self, msg: str) -> None:
        self.logger.warning(msg)

    def error(self, msg: str) -> None:
        self.logger.error(msg)


def pretty_print_config(config: dict[str, Any]) -> str:
    """Format configuration as an indented JSON string."""
    return json.dumps(config, indent=2, ensure_ascii=False, default=str)


def log_config(logger: Logger, config: dict[str, Any]) -> None:
    """Log the full merged configuration at training start."""
    logger.info("Configuration:\n" + pretty_print_config(config))
