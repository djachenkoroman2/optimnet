"""Настройка логирования: ход оптимизации пишется в файл optimnet.log
(каталог проекта) и дублируется в консоль."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

LOG_FILE = Path(__file__).resolve().parent.parent / "optimnet.log"

_FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def setup_logging(level: int = logging.INFO, to_console: bool = True) -> logging.Logger:
    """Создать логгер «optimnet» с файловым (optimnet.log) и консольным
    обработчиками; повторный вызов возвращает готовый логгер без дублирования."""
    logger = logging.getLogger("optimnet")
    if logger.handlers:  # повторный вызов не должен дублировать обработчики
        return logger
    logger.setLevel(level)

    fh = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    fh.setFormatter(logging.Formatter(_FMT))
    logger.addHandler(fh)

    if to_console:
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(logging.Formatter(_FMT))
        logger.addHandler(sh)

    logger.propagate = False
    logger.info("Логирование настроено: %s", LOG_FILE)
    return logger


def get_logger(name: str = "") -> logging.Logger:
    """Дочерний логгер пакета («optimnet.<name>», при пустом name — корневой)."""
    return logging.getLogger("optimnet" + (f".{name}" if name else ""))
