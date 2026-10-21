"""Вкладка «Задача»: выбор постановки (п. 5.1–5.3 и свёртка п. 6 из
doc/optim.md), веса α_k, Q_min, бюджет, Ψ_треб, перечень приоритетных пар."""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
    QHeaderView, QAbstractItemView,
)

from core.model import CriticalPair

from .state import AppState

FORMULATIONS = [
    ("max_survivability", "5.1 — максимизация Ψ при бюджете"),
    ("min_cost", "5.2 — минимизация стоимости при Ψ ≥ Ψ_треб"),
    ("minimax", "5.3 — робастная (минимакс по сценариям)"),
    ("scalarized", "6 — аддитивная свёртка критериев"),
]
QUALITY_MODES = [
    ("flow", "доля доставленного трафика (макс. поток)"),
    ("connectivity", "связность критических пар"),
]
VARIABLE_MODES = [
    ("discrete", "дискретный (выбор резервных каналов)"),
    ("continuous", "непрерывный (объёмы резервирования)"),
    ("mixed", "смешанный (каналы + объёмы)"),
]

CRITERIA_RU = ["Ψ (живучесть)", "C_Σ (стоимость)", "T_восст (время)", "Q̄ (среднее качество)"]


class ProblemTab(QWidget):
    """Вкладка «Задача»: выбор постановки (5.1–5.3 и свёртка п. 6 из
    doc/optim.md), режимов качества/переменных, параметров (Q_min, C_бюдж,
    Ψ_треб, t_ед, штраф M), весов свёртки α_k и приоритетных пар."""

    def __init__(self, state: AppState, parent=None):
        """Собрать формы постановки, параметров, весов и таблицу пар."""
        super().__init__(parent)
        self.state = state

        post = QGroupBox("Постановка задачи")
        form = QFormLayout(post)
        self.formulation = QComboBox()
        for key, label in FORMULATIONS:
            self.formulation.addItem(label, key)
        self.quality_mode = QComboBox()
        for key, label in QUALITY_MODES:
            self.quality_mode.addItem(label, key)
        self.var_mode = QComboBox()
        for key, label in VARIABLE_MODES:
            self.var_mode.addItem(label, key)
        form.addRow("постановка:", self.formulation)
        form.addRow("качество Q(x,s):", self.quality_mode)
        form.addRow("переменные:", self.var_mode)

        params = QGroupBox("Параметры")
        pf = QFormLayout(params)
        self.q_min = self._spin(pf, "Q_min (порог φ):", 0.0, 1.0, 0.8, 0.05)
        self.budget = self._spin(pf, "C_бюдж (бюджет):", 0.0, 1e9, 50.0, 1.0, decimals=2)
        self.psi_req = self._spin(pf, "Ψ_треб (для 5.2):", 0.0, 1.0, 0.9, 0.05)
        self.unit_time = self._spin(pf, "t_ед (время восст.):", 0.0, 1e6, 1.0, 0.5, decimals=2)
        self.penalty = self._spin(pf, "штраф M:", 0.0, 1e9, 10.0, 1.0, decimals=2)

        weights = QGroupBox("Веса свёртки α_k (п. 6)")
        wl = QHBoxLayout(weights)
        self.weights = []
        for name in CRITERIA_RU:
            box = QDoubleSpinBox()
            box.setRange(0.0, 1.0)
            box.setSingleStep(0.05)
            box.setDecimals(3)
            wl.addWidget(QLabel(name))
            wl.addWidget(box)
            self.weights.append(box)
        self.weights[0].setValue(0.5)
        self.weights[1].setValue(0.2)
        self.weights[2].setValue(0.1)
        self.weights[3].setValue(0.2)
        norm_btn = QPushButton("Нормировать Σα=1")
        norm_btn.clicked.connect(self._normalize_weights)
        wl.addWidget(norm_btn)

        pairs = QGroupBox("Приоритетные пары (a, b, r_ab) — ограничения λ_ab ≥ r_ab")
        pl = QVBoxLayout(pairs)
        self.pairs_table = QTableWidget(0, 3)
        self.pairs_table.setHorizontalHeaderLabels(["a", "b", "r_ab"])
        self.pairs_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.pairs_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.pairs_table.verticalHeader().setVisible(False)
        pl.addWidget(self.pairs_table)
        prow = QHBoxLayout()
        add_btn = QPushButton("Добавить пару")
        add_btn.clicked.connect(lambda: self._add_pair("", "", 1))
        del_btn = QPushButton("Удалить выбранные")
        del_btn.clicked.connect(self._remove_pairs)
        prow.addWidget(add_btn)
        prow.addWidget(del_btn)
        pl.addLayout(prow)

        apply_btn = QPushButton("Применить к конфигурации задачи")
        apply_btn.clicked.connect(self.apply_config)
        self.status = QLabel("")

        lay = QVBoxLayout(self)
        lay.addWidget(post)
        lay.addWidget(params)
        lay.addWidget(weights)
        lay.addWidget(pairs)
        lay.addWidget(apply_btn)
        lay.addWidget(self.status)
        lay.addStretch(1)
        self.setLayout(lay)
        self.refresh()

    @staticmethod
    def _spin(form, label, lo, hi, val, step, decimals=4):
        """Хелпер: поле ввода вещественного числа (QDoubleSpinBox) с диапазоном
        [lo, hi] и подписью, добавленное в форму form; вернуть виджет."""
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setSingleStep(step)
        s.setDecimals(decimals)
        s.setValue(val)
        form.addRow(label, s)
        return s

    # ------------------------------------------------------------- #

    def refresh(self) -> None:
        """Показать текущую конфигурацию state.config в элементах формы."""
        cfg = self.state.config
        self.formulation.setCurrentIndex(
            next((i for i in range(self.formulation.count())
                  if self.formulation.itemData(i) == cfg.formulation), 0))
        self.quality_mode.setCurrentIndex(
            next((i for i in range(self.quality_mode.count())
                  if self.quality_mode.itemData(i) == cfg.quality_mode), 0))
        self.var_mode.setCurrentIndex(
            next((i for i in range(self.var_mode.count())
                  if self.var_mode.itemData(i) == cfg.mode), 0))
        self.q_min.setValue(cfg.q_min)
        self.budget.setValue(cfg.budget)
        self.psi_req.setValue(cfg.psi_required)
        self.unit_time.setValue(cfg.unit_time)
        self.penalty.setValue(cfg.penalty)
        for w, val in zip(self.weights, cfg.weights):
            w.setValue(val)
        self.pairs_table.setRowCount(0)
        for p in self.state.critical_pairs:
            self._add_pair(p.a, p.b, p.r)
        self.status.setText(
            f"Текущая постановка: {dict(FORMULATIONS).get(cfg.formulation, cfg.formulation)}, "
            f"переменные: {cfg.mode}, Q_min={cfg.q_min}"
        )

    def _normalize_weights(self) -> None:
        """Привести веса α_k к сумме 1 (пропорционально текущим значениям)."""
        vals = [w.value() for w in self.weights]
        total = sum(vals)
        if total <= 0:
            return
        for w, v in zip(self.weights, vals):
            w.setValue(round(v / total, 3))

    def _add_pair(self, a: str, b: str, r: int) -> None:
        """Добавить строку (a, b, r_ab) в таблицу приоритетных пар."""
        row = self.pairs_table.rowCount()
        self.pairs_table.insertRow(row)
        self.pairs_table.setItem(row, 0, QTableWidgetItem(a))
        self.pairs_table.setItem(row, 1, QTableWidgetItem(b))
        self.pairs_table.setItem(row, 2, QTableWidgetItem(str(r)))

    def _remove_pairs(self) -> None:
        """Удалить выбранные строки таблицы пар."""
        rows = sorted({i.row() for i in self.pairs_table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.pairs_table.removeRow(r)

    def apply_config(self) -> None:
        """Забрать значения формы в state.config и state.critical_pairs;
        при неизвестных узлах в парах — оставить прежнюю конфигурацию
        и показать ошибку в статусе."""
        cfg = self.state.config
        cfg.formulation = self.formulation.currentData()
        cfg.quality_mode = self.quality_mode.currentData()
        cfg.mode = self.var_mode.currentData()
        cfg.q_min = self.q_min.value()
        cfg.budget = self.budget.value()
        cfg.psi_required = self.psi_req.value()
        cfg.unit_time = self.unit_time.value()
        cfg.penalty = self.penalty.value()
        cfg.weights = tuple(w.value() for w in self.weights)

        pairs: list[CriticalPair] = []
        net_nodes = set(self.state.network.nodes)
        for r in range(self.pairs_table.rowCount()):
            a = self.pairs_table.item(r, 0).text().strip()
            b = self.pairs_table.item(r, 1).text().strip()
            rtext = self.pairs_table.item(r, 2).text().strip()
            if not a or not b:
                continue
            if a not in net_nodes or b not in net_nodes:
                self.status.setText(f"Ошибка: пара ({a}, {b}) — неизвестные узлы")
                return
            pairs.append(CriticalPair(a, b, int(rtext or 1)))
        self.state.critical_pairs = pairs
        self.status.setText(
            f"Применено: постановка {cfg.formulation}, переменные {cfg.mode}, "
            f"пар приоритетных: {len(pairs)}"
        )
