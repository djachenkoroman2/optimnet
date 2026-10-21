"""Скриншоты GUI для руководства пользователя (doc/images/).

Запуск (offscreen, без показа окна):
    .venv/Scripts/python.exe scripts/make_screenshots.py

Перед импортом PyQt6 включается платформа «offscreen». Главное окно
загружает пример medium.json, прогоняются ga/sa/greedy (seeds 42..44),
вкладки «Эксперимент» и «Результаты» заполняются, затем каждая вкладка
снимается методом widget.grab().save(...).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"  # до импорта PyQt6
# платформа offscreen не подхватывает системные шрифты Windows сама —
# без этого кириллица рисуется пустыми квадратиками (0 семейств в БД)
os.environ.setdefault("QT_QPA_FONTDIR", r"C:\Windows\Fonts")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

IMAGES = ROOT / "doc" / "images"
IMAGES.mkdir(parents=True, exist_ok=True)

from PyQt6.QtWidgets import QApplication, QMessageBox, QTabWidget  # noqa: E402
from PyQt6.QtGui import QFont  # noqa: E402

from core.problem import Problem  # noqa: E402

from algorithms.registry import OPTIMIZERS  # noqa: E402

from gui.main_window import MainWindow  # noqa: E402
from gui.workers import _run_timed  # noqa: E402

SPECS = {s.key: s for s in OPTIMIZERS}

ABOUT_TEXT = (
    "OptimNet — ПО оптимизации живучести компьютерной сети.\n\n"
    "Сеть моделируется взвешенным ориентированным графом G = ⟨V, E, C, W⟩; "
    "живучесть Ψ(x) = Σ p(s)·φ(Q(x,s)) (doc/optim.md). Постановки 5.1–5.3 "
    "и аддитивная свёртка критериев; алгоритмы: генетический, имитация "
    "отжига, ветви и границы, жадная эвристика.\n\n"
    "Версия 1.0, Python + PyQt6."
)


def shoot(widget, name: str, app: QApplication) -> Path:
    app.processEvents()
    app.processEvents()
    path = IMAGES / name
    if not widget.grab().save(str(path)):
        raise RuntimeError(f"Не удалось сохранить снимок: {path}")
    return path


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("OptimNet")
    app.setFont(QFont("Segoe UI", 9))

    win = MainWindow()
    # в конструкторе MainWindow данные грузятся до создания вкладок и
    # refresh() не вызывается — как при загрузке примера через меню «Файл»
    win._refresh_all()
    win.resize(1280, 860)
    win.show()
    app.processEvents()

    shots: list[Path] = []

    # ---- 01: «Топология» с графом medium ------------------------------
    win.tabs.setCurrentIndex(0)
    shots.append(shoot(win.tabs.widget(0), "01_main_window_topology.png", app))

    # ---- 02–05: внутренние вкладки таблиц -----------------------------
    inner = win.topology_tab.findChild(QTabWidget)
    assert inner is not None, "В TopologyTab не найден дочерний QTabWidget"
    for idx, name in enumerate(
        ["02_topology_nodes.png", "03_topology_links.png",
         "04_topology_demands.png", "05_topology_candidates.png"]
    ):
        inner.setCurrentIndex(idx)
        shots.append(shoot(win.topology_tab, name, app))

    # ---- 06: «Сценарии» ------------------------------------------------
    win.tabs.setCurrentIndex(1)
    shots.append(shoot(win.tabs.widget(1), "06_tab_scenarios.png", app))

    # ---- 07: «Задача», 08: «Алгоритм» ----------------------------------
    win.tabs.setCurrentIndex(2)
    shots.append(shoot(win.tabs.widget(2), "07_tab_problem.png", app))
    win.tabs.setCurrentIndex(3)
    shots.append(shoot(win.tabs.widget(3), "08_tab_algorithm.png", app))

    # ---- прогоны ga/sa/greedy (синхронно, seeds 42..44) ----------------
    problem = Problem(win.state.network, win.state.scenarios,
                      win.state.config, win.state.critical_pairs)
    seeds = [42, 43, 44]
    full_results: dict[str, list] = {}
    for key in ("ga", "sa", "greedy"):
        runs = [_run_timed(SPECS[key], problem, {}, None, s) for s in seeds]
        full_results[key] = runs
        best = max(runs, key=lambda r: r.eval_result.objective)
        win.state.last_results[key] = best
        print(f"{key}: лучший F = {best.eval_result.objective:.6f} "
              f"(seed={seeds[runs.index(best)]})")

    # ---- 09: «Эксперимент» — таблица и график сходимости ---------------
    win.experiment_tab.worker_full_results = full_results
    win.experiment_tab.table.setRowCount(0)
    for key in ("ga", "sa", "greedy"):
        win.experiment_tab._fill_row(key, win.state.last_results[key])
    win.experiment_tab.plot.plot_histories(
        {k: v.history for k, v in win.state.last_results.items()})
    win.experiment_tab.progress.setValue(100)
    win.experiment_tab.progress_label.setText("Эксперимент завершён")
    win.tabs.setCurrentIndex(4)
    shots.append(shoot(win.tabs.widget(4), "09_tab_experiment.png", app))

    # ---- 10: «Результаты» ----------------------------------------------
    win.results_tab.refresh()
    win.tabs.setCurrentIndex(5)
    shots.append(shoot(win.tabs.widget(5), "10_tab_results.png", app))

    # ---- 11: «Топология» с решением ГА на графе ------------------------
    best_ga = win.state.last_results["ga"]
    win.tabs.setCurrentIndex(0)
    win.topology_tab.canvas.clear_positions()
    win.topology_tab.canvas.draw_network(
        win.state.network,
        scenario=None,
        solution=best_ga.best,
        title="Резервные каналы — Генетический алгоритм",
    )
    shots.append(shoot(win.tabs.widget(0), "11_graph_solution.png", app))

    # ---- 12: диалог «О программе» --------------------------------------
    msg = QMessageBox(win)
    msg.setWindowTitle("О программе")
    msg.setText(ABOUT_TEXT)
    msg.show()
    app.processEvents()
    app.processEvents()
    path = IMAGES / "12_dialog_about.png"
    if not msg.grab().save(str(path)):
        raise RuntimeError(f"Не удалось сохранить снимок: {path}")
    msg.close()
    shots.append(path)

    # ---- 13: «Сценарии» — редактор второго сценария --------------------
    win.tabs.setCurrentIndex(1)
    win.scenarios_tab.list.setCurrentRow(1)
    shots.append(shoot(win.tabs.widget(1), "13_scenario_editor.png", app))

    # ---- проверка PNG ---------------------------------------------------
    print("\nСнимки:")
    ok = True
    for p in shots:
        data = p.read_bytes()
        valid = len(data) > 0 and data[:8] == b"\x89PNG\r\n\x1a\n"
        ok &= valid
        print(f"  {p.name:34s} {len(data):>8d} байт  {'OK' if valid else 'НЕВАЛИДЕН'}")
    if not ok:
        return 1
    print(f"\nВсе {len(shots)} снимков сохранены в {IMAGES}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
