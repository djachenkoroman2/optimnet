"""Вкладка «Результаты»: найденное решение x, значения критериев
(Ψ, C_Σ, T_восст, Q̄, Q_min), проверка ограничений, выделение резервных
каналов на графе, экспорт в CSV/JSON."""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QComboBox, QFileDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget, QHeaderView,
    QAbstractItemView,
)

from algorithms.base import OptimizeResult
from core import metrics
from core.io import export_result_csv, save_json

from .state import AppState


def result_to_dict(res: OptimizeResult, state: AppState) -> dict:
    """Сериализовать результат прогона в JSON-совместимый словарь: критерии
    (Ψ, C_Σ, T_восст, Q̄, min Q), нарушения, выбранные кандидаты с объёмами
    и качество Q(x, s) с φ(Q) по каждому сценарию."""
    net, cfg = state.network, state.config
    er = res.eval_result
    vols = res.best.volumes()
    x = [
        {
            "candidate": c.id,
            "source": c.source,
            "target": c.target,
            "volume": round(float(v), 4),
            "capacity_added": round(float(c.capacity * v), 4),
            "cost": round(float(c.cost * v), 4),
        }
        for c, v in zip(net.candidates, vols) if v > 0
    ]
    per_scenario = [
        {
            "scenario": s.id,
            "p": s.p,
            "Q": round(metrics.quality(net, res.best, s, cfg.quality_mode), 6),
            "phi": int(metrics.phi(metrics.quality(net, res.best, s, cfg.quality_mode), cfg.q_min)),
        }
        for s in state.scenarios
    ]
    return {
        "algorithm": res.algorithm,
        "objective": round(er.objective, 6),
        "psi": round(er.psi, 6),
        "cost": round(er.cost, 6),
        "t_recovery": round(er.t_rec, 6),
        "q_mean": round(er.q_mean, 6),
        "q_min_actual": round(er.q_min_actual, 6),
        "feasible": er.feasible,
        "violations": er.violations,
        "iterations": res.iterations,
        "evaluations": res.evaluations,
        "runtime_sec": round(res.runtime, 4),
        "solution": x,
        "per_scenario": per_scenario,
    }


class ResultsTab(QWidget):
    """Вкладка «Результаты»: выбор прогона, сводка критериев и допустимости,
    таблица решения x, таблица Q(x, s)/φ(Q) по сценариям, показ резерва
    на графе и экспорт в CSV/JSON."""

    def __init__(self, state: AppState, parent=None):
        """Собрать селектор результата, сводку, две таблицы и кнопки."""
        super().__init__(parent)
        self.state = state

        top = QHBoxLayout()
        top.addWidget(QLabel("Результат алгоритма:"))
        self.selector = QComboBox()
        self.selector.currentIndexChanged.connect(self.refresh_view)
        top.addWidget(self.selector)
        top.addStretch(1)

        self.summary = QLabel("Решений пока нет — запустите эксперимент.")
        self.summary.setWordWrap(True)

        self.x_table = QTableWidget(0, 5)
        self.x_table.setHorizontalHeaderLabels(["кандидат", "источник", "назначение", "объём", "стоимость"])
        self.x_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.x_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.x_table.verticalHeader().setVisible(False)

        self.scen_table = QTableWidget(0, 4)
        self.scen_table.setHorizontalHeaderLabels(["сценарий", "p(s)", "Q(x,s)", "φ(Q)"])
        self.scen_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.scen_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.scen_table.verticalHeader().setVisible(False)

        btn_row = QHBoxLayout()
        show_btn = QPushButton("Показать решение на графе")
        show_btn.clicked.connect(self.show_on_graph)
        export_csv_btn = QPushButton("Экспорт CSV…")
        export_csv_btn.clicked.connect(self.export_csv)
        export_json_btn = QPushButton("Экспорт JSON…")
        export_json_btn.clicked.connect(self.export_json)
        btn_row.addWidget(show_btn)
        btn_row.addWidget(export_csv_btn)
        btn_row.addWidget(export_json_btn)
        btn_row.addStretch(1)

        lay = QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(self.summary)
        lay.addWidget(QLabel("Вектор решения x (выбранные резервные каналы):"))
        lay.addWidget(self.x_table)
        lay.addWidget(QLabel("Качество по сценариям:"))
        lay.addWidget(self.scen_table)
        lay.addLayout(btn_row)
        self.setLayout(lay)

    # ------------------------------------------------------------- #

    def refresh(self) -> None:
        """Перестроить селектор по state.last_results (сохранив выбор) и
        обновить отображение."""
        current = self.selector.currentData()
        self.selector.blockSignals(True)
        self.selector.clear()
        for key, res in self.state.last_results.items():
            self.selector.addItem(f"{res.algorithm} (F={res.eval_result.objective:.4f})", key)
        if self.state.last_results:
            idx = next((i for i in range(self.selector.count())
                        if self.selector.itemData(i) == current), 0)
            self.selector.setCurrentIndex(idx)
        self.selector.blockSignals(False)
        self.refresh_view()

    def current_result(self) -> OptimizeResult | None:
        """Результат, выбранный в селекторе (или None)."""
        key = self.selector.currentData()
        return self.state.last_results.get(key)

    def refresh_view(self) -> None:
        """Заполнить сводку и таблицы по выбранному результату; пустое
        состояние — при отсутствии результатов."""
        res = self.current_result()
        self.x_table.setRowCount(0)
        self.scen_table.setRowCount(0)
        if res is None:
            self.summary.setText("Решений пока нет — запустите эксперимент.")
            return
        er = res.eval_result
        viol = ", ".join(f"{k}={v:.4f}" for k, v in er.violations.items()) or "нет"
        self.summary.setText(
            f"Алгоритм: {res.algorithm}  |  F = {er.objective:.6f}  |  "
            f"Ψ = {er.psi:.4f}  |  C_Σ = {er.cost:.2f}  |  T_восст = {er.t_rec:.4f}  |  "
            f"Q̄ = {er.q_mean:.4f}  |  min Q = {er.q_min_actual:.4f}  |  "
            f"допустимо: {'да' if er.feasible else 'НЕТ'}  |  нарушения: {viol}  |  "
            f"итераций: {res.iterations}, оценок: {res.evaluations}, время: {res.runtime:.3f} с"
        )
        vols = res.best.volumes()
        for c, v in zip(self.state.network.candidates, vols):
            if v <= 0:
                continue
            row = self.x_table.rowCount()
            self.x_table.insertRow(row)
            for col, val in enumerate([c.id, c.source, c.target, f"{v:.3f}", f"{c.cost * v:.2f}"]):
                self.x_table.setItem(row, col, QTableWidgetItem(str(val)))
        cfg = self.state.config
        for s in self.state.scenarios:
            q = metrics.quality(self.state.network, res.best, s, cfg.quality_mode)
            row = self.scen_table.rowCount()
            self.scen_table.insertRow(row)
            for col, val in enumerate([s.id, f"{s.p:.4f}", f"{q:.4f}", f"{metrics.phi(q, cfg.q_min):.0f}"]):
                self.scen_table.setItem(row, col, QTableWidgetItem(str(val)))

    def show_on_graph(self) -> None:
        """Перейти на вкладку «Топология» и показать на графе резервные каналы
        выбранного решения."""
        res = self.current_result()
        if res is None:
            QMessageBox.information(self, "Нет решения", "Сначала запустите оптимизацию.")
            return
        win = self.window()
        if hasattr(win, "topology_tab"):
            win.topology_tab.canvas.clear_positions()
            win.topology_tab.canvas.draw_network(
                self.state.network,
                scenario=None,
                solution=res.best,
                title=f"Резервные каналы — {res.algorithm}",
            )
            win.tabs.setCurrentIndex(0)

    def export_csv(self) -> None:
        """Экспортировать критерии, компоненты решения и Q по сценариям
        выбранного результата в CSV."""
        res = self.current_result()
        if res is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Экспорт результата", "result.csv", "CSV (*.csv)")
        if not path:
            return
        d = result_to_dict(res, self.state)
        rows = [
            {"показатель": "objective", "значение": d["objective"]},
            {"показатель": "psi", "значение": d["psi"]},
            {"показатель": "cost", "значение": d["cost"]},
            {"показатель": "t_recovery", "значение": d["t_recovery"]},
            {"показатель": "q_mean", "значение": d["q_mean"]},
            {"показатель": "q_min_actual", "значение": d["q_min_actual"]},
            {"показатель": "feasible", "значение": d["feasible"]},
        ] + [
            {"показатель": f"x[{x['candidate']}]", "значение": x["volume"]} for x in d["solution"]
        ] + [
            {"показатель": f"Q[{s['scenario']}]", "значение": s["Q"]} for s in d["per_scenario"]
        ]
        export_result_csv(rows, path)

    def export_json(self) -> None:
        """Экспортировать выбранный результат в JSON (result_to_dict)."""
        res = self.current_result()
        if res is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Экспорт результата", "result.json", "JSON (*.json)")
        if path:
            save_json(result_to_dict(res, self.state), path)
