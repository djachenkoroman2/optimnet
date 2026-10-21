"""Вкладка «Эксперимент»: запуск нескольких алгоритмов на одном наборе
данных, сравнительная таблица (лучшее/среднее/худшее, время, итерации)
и график сходимости (matplotlib, встроенный в окно)."""

from __future__ import annotations

import numpy as np
from PyQt6.QtWidgets import (
    QCheckBox, QFileDialog, QHBoxLayout, QLabel, QMessageBox, QProgressBar,
    QPushButton, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
    QHeaderView, QAbstractItemView,
)

from core.io import export_result_csv, save_json
from core.logging_setup import get_logger
from core.problem import Problem

from algorithms.base import OptimizeResult
from algorithms.registry import available_optimizers

from .state import AppState
from .workers import ExperimentWorker

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as MplCanvas


class ConvergenceCanvas(MplCanvas):
    """Холст графика сходимости: лучшее значение целевой функции по итерациям
    для каждого алгоритма."""

    def __init__(self, parent=None):
        """Пустой холст с подписанными осями (итерация / лучшее значение)."""
        self.fig = Figure(figsize=(5, 3.4), tight_layout=True)
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("итерация")
        self.ax.set_ylabel("лучшее значение")

    def plot_histories(self, histories: dict[str, list[float]]) -> None:
        """Построить кривые сходимости: histories — {имя алгоритма: history}."""
        self.ax.clear()
        self.ax.set_xlabel("итерация")
        self.ax.set_ylabel("лучшее значение")
        self.ax.grid(True, alpha=0.3)
        for name, hist in histories.items():
            if hist:
                self.ax.plot(range(1, len(hist) + 1), hist, label=name, linewidth=1.4)
        if histories:
            self.ax.legend(fontsize=8)
        self.draw_idle()


class ExperimentTab(QWidget):
    """Вкладка «Эксперимент»: запуск нескольких алгоритмов по нескольку
    прогонов (разные seed) в фоновом потоке, сравнительная таблица
    (лучшее/среднее/худшее F, время, итерации) и график сходимости."""

    def __init__(self, state: AppState, parent=None):
        """Собрать панель настроек, кнопки, прогресс, таблицу и график."""
        super().__init__(parent)
        self.state = state
        self.log = get_logger("gui.experiment")
        self.worker: ExperimentWorker | None = None  # фоновый поток эксперимента
        self.algorithm_checks: dict[str, QCheckBox] = {}
        self.worker_full_results: dict[str, list] = {}  # все прогоны по алгоритмам

        top = QHBoxLayout()
        top.addWidget(QLabel("Число прогонов на алгоритм:"))
        self.runs_spin = QSpinBox()
        self.runs_spin.setRange(1, 50)
        self.runs_spin.setValue(3)
        top.addWidget(self.runs_spin)
        top.addWidget(QLabel("seed (базовый):"))
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999999)
        self.seed_spin.setValue(42)
        top.addWidget(self.seed_spin)
        top.addStretch(1)

        self.checks_row = QHBoxLayout()
        self._rebuild_checks()

        self.run_btn = QPushButton("Запустить эксперимент")
        self.run_btn.clicked.connect(self.start_experiment)
        self.stop_btn = QPushButton("Остановить")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_experiment)
        self.export_btn = QPushButton("Экспорт таблицы (CSV)…")
        self.export_btn.clicked.connect(self.export_csv)
        self.export_json_btn = QPushButton("Экспорт JSON…")
        self.export_json_btn.clicked.connect(self.export_json)

        btns = QHBoxLayout()
        btns.addWidget(self.run_btn)
        btns.addWidget(self.stop_btn)
        btns.addWidget(self.export_btn)
        btns.addWidget(self.export_json_btn)
        btns.addStretch(1)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress_label = QLabel("Готов")

        self.table = QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels([
            "алгоритм", "лучшее F", "среднее F", "худшее F",
            "Ψ лучшего", "C_Σ лучшего", "время, с (среднее)", "итераций (среднее)", "оценок F (новых)",
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)

        self.plot = ConvergenceCanvas()

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.addWidget(self.plot)
        rl.addWidget(QLabel("График сходимости (лучший прогон каждого алгоритма)"))

        bottom = QHBoxLayout()
        bottom.addWidget(self.table, 2)
        bottom.addWidget(right, 1)

        lay = QVBoxLayout(self)
        lay.addLayout(top)
        lay.addLayout(self.checks_row)
        lay.addLayout(btns)
        lay.addWidget(self.progress)
        lay.addWidget(self.progress_label)
        lay.addLayout(bottom, 1)
        self.setLayout(lay)

    # ------------------------------------------------------------- #

    def refresh(self) -> None:
        """Перестроить галочки алгоритмов под текущую постановку/режим."""
        self._rebuild_checks()

    def _rebuild_checks(self) -> None:
        """Создать QCheckBox для каждого совместимого алгоритма (все включены)."""
        while self.checks_row.count():
            item = self.checks_row.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.algorithm_checks = {}
        cfg = self.state.config
        for spec in available_optimizers(cfg.formulation, cfg.mode):
            cb = QCheckBox(spec.name)
            cb.setChecked(True)
            self.algorithm_checks[spec.key] = cb
            self.checks_row.addWidget(cb)
        self.checks_row.addStretch(1)

    def _selected_specs(self):
        """Спецификации отмеченных галочками алгоритмов (в порядке реестра)."""
        cfg = self.state.config
        specs = {s.key: s for s in available_optimizers(cfg.formulation, cfg.mode)}
        return [specs[k] for k in self.algorithm_checks if self.algorithm_checks[k].isChecked() and k in specs]

    # ------------------------------------------------------------- #

    def start_experiment(self) -> None:
        """Проверить данные и выбор, собрать Problem, seed'ы и параметры,
        запустить ExperimentWorker в фоне и заблокировать кнопку запуска."""
        if not self.state.has_data():
            QMessageBox.warning(self, "Нет данных",
                                "Загрузите пример и задайте сценарии перед запуском.")
            return
        specs = self._selected_specs()
        if not specs:
            QMessageBox.warning(self, "Нет алгоритмов", "Отметьте хотя бы один алгоритм.")
            return

        problem = Problem(self.state.network, self.state.scenarios,
                          self.state.config, self.state.critical_pairs)
        n_runs = self.runs_spin.value()
        base_seed = self.seed_spin.value()
        seeds = {s.key: [base_seed + i for i in range(n_runs)] for s in specs}
        params = self._params_from_tabs(specs)

        self.worker = ExperimentWorker(specs, problem, params, seeds, parent=self)
        self.worker.progressed.connect(self._on_progress)
        self.worker.algorithm_done.connect(self._on_algorithm_done)
        self.worker.finished_ok.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress.setValue(0)
        self.progress_label.setText("Эксперимент запущен…")
        self.log.info("Эксперимент: алгоритмов=%d, прогонов=%d", len(specs), n_runs)

    def _params_from_tabs(self, specs) -> dict[str, dict]:
        """Параметры — из вкладки «Алгоритм» (для выбранного там) либо
        значения по умолчанию для остальных."""
        params: dict[str, dict] = {}
        main = self.window().algorithm_tab.current_params() if hasattr(self.window(), "algorithm_tab") else {}
        main_spec = self.window().algorithm_tab.current_spec() if hasattr(self.window(), "algorithm_tab") else None
        for spec in specs:
            if main_spec is not None and spec.key == main_spec.key:
                params[spec.key] = dict(main)
            else:
                params[spec.key] = {p.name: p.default for p in spec.params}
        return params

    def stop_experiment(self) -> None:
        """Мягко остановить фоновый эксперимент (после текущей итерации)."""
        if self.worker:
            self.worker.stop()
            self.progress_label.setText("Остановка после текущей итерации…")

    # ------------------------------------------------------------- #

    def _on_progress(self, key: str, it: int, best: float) -> None:
        """Подпись прогресса по активному алгоритму; бар двигаем грубо —
        по числу завершённых алгоритмов (общего числа итераций нет)."""
        self.progress_label.setText(f"[{key}] итерация {it}, лучшее F = {best:.6f}")
        # без общего числа итераций прогресс «по кругу» не делаем: просто
        # отражаем активность через подпись; бар двигаем по алгоритмам
        done = sum(1 for k, v in self.algorithm_checks.items()
                   if self.worker and k in getattr(self.worker, "_done_keys", set()))
        self.progress.setValue(min(99, done * 30))

    def _on_algorithm_done(self, key: str, best: OptimizeResult) -> None:
        """Запомнить лучший прогон алгоритма, обновить его строку таблицы
        и график сходимости."""
        self.state.last_results[key] = best
        if not hasattr(self.worker, "_done_keys"):
            self.worker._done_keys = set()
        self.worker._done_keys.add(key)
        self._fill_row(key, best)
        self.plot.plot_histories({k: v.history for k, v in self.state.last_results.items()})

    def _fill_row(self, key: str, best: OptimizeResult) -> None:
        """Заполнить/обновить строку таблицы по всем прогонам алгоритма."""
        cfg = self.state.config
        specs = {s.key: s for s in available_optimizers(cfg.formulation, cfg.mode)}
        name = specs[key].name if key in specs else key
        all_results = self.worker_full_results.get(key, [best])
        objs = [r.eval_result.objective for r in all_results]
        times = [r.runtime for r in all_results]
        iters = [r.iterations for r in all_results]
        evals = [r.evaluations for r in all_results]

        def find_row() -> int:
            for r in range(self.table.rowCount()):
                if self.table.item(r, 0) and self.table.item(r, 0).text() == name:
                    return r
            return -1

        row = find_row()
        if row < 0:
            row = self.table.rowCount()
            self.table.insertRow(row)
        vals = [
            name,
            f"{max(objs):.6f}", f"{float(np.mean(objs)):.6f}", f"{min(objs):.6f}",
            f"{best.eval_result.psi:.4f}",
            f"{best.eval_result.cost:.2f}",
            f"{float(np.mean(times)):.3f}",
            f"{float(np.mean(iters)):.1f}",
            f"{int(np.mean(evals))}",
        ]
        for c, v in enumerate(vals):
            self.table.setItem(row, c, QTableWidgetItem(v))

    def _on_finished(self, results: dict) -> None:
        """Финал: сохранить все прогоны, перезаполнить таблицу по полным
        данным, довести прогресс до 100% и обновить смежные вкладки."""
        self.worker_full_results = results
        # перезаполняем таблицу целиком по полным данным
        self.table.setRowCount(0)
        for key, runs in results.items():
            if not runs:
                continue
            best = max(runs, key=lambda r: r.eval_result.objective)
            self.state.last_results[key] = best
            self._fill_row(key, best)
        self.plot.plot_histories({k: v.history for k, v in self.state.last_results.items()})
        self.progress.setValue(100)
        self.progress_label.setText("Эксперимент завершён")
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        if hasattr(self.window(), "results_tab"):
            self.window().results_tab.refresh()
        if hasattr(self.window(), "topology_tab"):
            self.window().topology_tab.redraw_graph()

    def _on_failed(self, message: str) -> None:
        """Показать ошибку эксперимента и разблокировать кнопки."""
        QMessageBox.critical(self, "Ошибка эксперимента", message)
        self.progress_label.setText("Ошибка")
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    # ------------------------------------------------------------- #

    def export_csv(self) -> None:
        """Экспортировать сравнительную таблицу в CSV."""
        rows = []
        for r in range(self.table.rowCount()):
            rows.append({
                self.table.horizontalHeaderItem(c).text(): self.table.item(r, c).text()
                for c in range(self.table.columnCount())
            })
        if not rows:
            QMessageBox.information(self, "Пусто", "Таблица результатов пуста.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Экспорт таблицы", "experiment.csv", "CSV (*.csv)")
        if path:
            export_result_csv(rows, path)

    def export_json(self) -> None:
        """Экспортировать конфигурацию и лучшие результаты всех алгоритмов
        в JSON."""
        if not self.state.last_results:
            QMessageBox.information(self, "Пусто", "Нет результатов для экспорта.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Экспорт результатов", "results.json", "JSON (*.json)")
        if not path:
            return
        from gui.tab_results import result_to_dict
        data = {
            "config": {
                "formulation": self.state.config.formulation,
                "q_min": self.state.config.q_min,
                "budget": self.state.config.budget,
                "mode": self.state.config.mode,
                "quality_mode": self.state.config.quality_mode,
                "weights": list(self.state.config.weights),
            },
            "algorithms": {
                key: result_to_dict(res, self.state) for key, res in self.state.last_results.items()
            },
        }
        save_json(data, path)
