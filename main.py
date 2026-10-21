"""Точка входа: настройка логирования и запуск GUI."""

from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

from core.logging_setup import setup_logging
from gui.main_window import MainWindow


def main() -> int:
    """Настроить логирование, создать приложение Qt и показать главное окно;
    вернуть код выхода цикла событий."""
    setup_logging()
    app = QApplication(sys.argv)
    app.setApplicationName("OptimNet")
    app.setOrganizationName("OptimNet")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
