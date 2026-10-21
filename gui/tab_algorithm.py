"""Вкладка «Алгоритм»: выбор оптимизатора и его параметров. Форма строится
автоматически из реестра (algorithms/registry.py) — добавление нового
алгоритма не требует правок в UI."""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QLabel, QSpinBox,
    QVBoxLayout, QWidget,
)

from algorithms.registry import available_optimizers

from .state import AppState


class AlgorithmTab(QWidget):
    """Вкладка выбора алгоритма и его параметров.

    Список алгоритмов и поля параметров строятся автоматически из реестра
    (algorithms/registry.py) под текущую постановку/режим; виджеты параметров
    хранятся в param_widgets {имя: QSpinBox/QDoubleSpinBox}.
    """

    def __init__(self, state: AppState, parent=None):
        """Собрать селектор алгоритма, группу параметров и подсказку."""
        super().__init__(parent)
        self.state = state

        self.selector = QComboBox()
        self.selector.currentIndexChanged.connect(self._rebuild_params)

        self.params_box = QGroupBox("Параметры алгоритма")
        self.params_form = QFormLayout(self.params_box)
        self.param_widgets: dict = {}

        self.hint = QLabel("")
        self.hint.setWordWrap(True)

        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Алгоритм решения:"))
        lay.addWidget(self.selector)
        lay.addWidget(self.params_box)
        lay.addWidget(self.hint)
        lay.addStretch(1)
        self.setLayout(lay)
        self.refresh()

    # ------------------------------------------------------------- #

    def specs(self):
        """Спецификации алгоритмов, совместимых с текущей постановкой/режимом."""
        cfg = self.state.config
        return available_optimizers(cfg.formulation, cfg.mode)

    def refresh(self) -> None:
        """Перестроить список алгоритмов (сохранив выбор, если он ещё есть)
        и форму параметров; предупредить, если алгоритмов нет."""
        current_key = self.selector.currentData()
        self.selector.blockSignals(True)
        self.selector.clear()
        specs = self.specs()
        for spec in specs:
            self.selector.addItem(spec.name, spec.key)
        if specs:
            idx = next((i for i in range(self.selector.count())
                        if self.selector.itemData(i) == current_key), 0)
            self.selector.setCurrentIndex(idx)
        self.selector.blockSignals(False)
        self._rebuild_params()
        if not specs:
            self.hint.setText("Нет алгоритмов, совместимых с текущей постановкой. "
                              "Измените постановку или режим переменных.")

    def current_spec(self):
        """Спецификация выбранного в селекторе алгоритма (или None)."""
        key = self.selector.currentData()
        return next((s for s in self.specs() if s.key == key), None)

    def current_params(self) -> dict:
        """Значения параметров из виджетов формы: {имя: число}."""
        return {name: w.value() for name, w in self.param_widgets.items()}

    def _rebuild_params(self) -> None:
        """Заполнить форму виджетами параметров выбранного алгоритма
        (целые — QSpinBox, вещественные — QDoubleSpinBox по спецификации)."""
        while self.params_form.rowCount():
            self.params_form.removeRow(0)
        self.param_widgets = {}
        spec = self.current_spec()
        if spec is None:
            return
        for p in spec.params:
            if p.kind == "int":
                w = QSpinBox()
                w.setRange(int(p.lo), int(p.hi))
                w.setSingleStep(max(1, int(p.step)))
                w.setValue(int(p.default))
            else:
                w = QDoubleSpinBox()
                w.setRange(p.lo, p.hi)
                w.setSingleStep(p.step)
                w.setDecimals(4)
                w.setValue(p.default)
            self.param_widgets[p.name] = w
            self.params_form.addRow(p.label + ":", w)
        self.hint.setText(
            f"Алгоритм «{spec.name}». Параметры подобраны автоматически из реестра; "
            "новые алгоритмы добавляются без изменения кода интерфейса."
        )
