import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from lucidtree.common.paths import get_project_root


def setup_logger(
    name: str, log_file: Path | str, level: int = logging.INFO
) -> logging.Logger:
    """
    Set up a logger instance

    Args:
        name (str): the name of the logger
        log_file (Path | str): the name of the .log file
        level (int, optional): the default logger level. Defaults to logging.INFO.

    Returns:
        logging.Logger: the logger
    """
    root = get_project_root() / "logs"
    path = root / log_file
    path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(path, maxBytes=10_000_000, backupCount=5)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
