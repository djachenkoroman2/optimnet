"""Вкладка «Сценарии»: список сценариев поражения, ручной редактор
(отказавшие узлы/каналы, вероятность) и генератор случайных отказов."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QMessageBox, QPushButton, QSpinBox,
    QSplitter, QVBoxLayout, QWidget,
)

from core.io import generate_scenarios, load_scenarios, load_scenarios_csv, save_scenarios, save_scenarios_csv
from core.model import Scenario

from .state import AppState


def _fmt_set(nodes: set[str]) -> str:
    """Множество имён узлов → строка «a, b, c» для редактора."""
    return ", ".join(sorted(nodes))


def _fmt_links(links: set[tuple[str, str]]) -> str:
    """Множество пар каналов → строка «a-b, c-d» для редактора."""
    return ", ".join(f"{a}-{b}" for a, b in sorted(links))


def _parse_nodes(text: str) -> set[str]:
    """Обратное _fmt_set: строка через запятую/точку с запятой → множество узлов."""
    return {x.strip() for x in text.replace(";", ",").split(",") if x.strip()}


def _parse_links(text: str) -> set[tuple[str, str]]:
    """Обратное _fmt_links: «a-b, c-d» → множество пар; элемент без «-»
    считается ошибкой формата (ValueError)."""
    out: set[tuple[str, str]] = set()
    for item in text.replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        if "-" not in item:
            raise ValueError(f"Канал '{item}': ожидается формат ИСТОЧНИК-НАЗНАЧЕНИЕ")
        a, _, b = item.partition("-")
        out.add((a.strip(), b.strip()))
    return out


class ScenariosTab(QWidget):
    """Вкладка «Сценарии»: список сценариев поражения с ручным редактором
    (id, p(s), отказавшие узлы/каналы), генератор случайных отказов и
    загрузка/сохранение JSON/CSV."""

    def __init__(self, state: AppState, parent=None):
        """Собрать список сценариев, редактор, блок генератора и кнопки файлов."""
        super().__init__(parent)
        self.state = state
        self._loading = False  # флаг: идёт программное обновление списка

        self.list = QListWidget()
        self.list.currentRowChanged.connect(self._on_select)

        editor = QGroupBox("Редактор сценария")
        form = QFormLayout(editor)
        self.ed_id = QLineEdit()
        self.ed_p = QDoubleSpinBox()
        self.ed_p.setRange(0.0, 1.0)
        self.ed_p.setSingleStep(0.05)
        self.ed_p.setDecimals(4)
        self.ed_nodes = QLineEdit()
        self.ed_nodes.setPlaceholderText("R2, S1 — через запятую")
        self.ed_links = QLineEdit()
        self.ed_links.setPlaceholderText("R1-R2, S1-R3 — ИСТОЧНИК-НАЗНАЧЕНИЕ")
        form.addRow("id:", self.ed_id)
        form.addRow("вероятность p(s):", self.ed_p)
        form.addRow("отказавшие узлы:", self.ed_nodes)
        form.addRow("отказавшие каналы:", self.ed_links)

        btn_row = QHBoxLayout()
        apply_btn = QPushButton("Применить изменения")
        apply_btn.clicked.connect(self._apply_editor)
        add_btn = QPushButton("Новый сценарий")
        add_btn.clicked.connect(self._add_scenario)
        del_btn = QPushButton("Удалить")
        del_btn.clicked.connect(self._delete_selected)
        btn_row.addWidget(apply_btn)
        btn_row.addWidget(add_btn)
        btn_row.addWidget(del_btn)

        # генератор
        gen = QGroupBox("Генератор случайных отказов")
        gen_form = QFormLayout(gen)
        self.gen_count = QSpinBox(); self.gen_count.setRange(1, 500); self.gen_count.setValue(8)
        self.gen_p_link = QDoubleSpinBox(); self.gen_p_link.setRange(0.0, 1.0); self.gen_p_link.setValue(0.15); self.gen_p_link.setSingleStep(0.01)
        self.gen_p_node = QDoubleSpinBox(); self.gen_p_node.setRange(0.0, 1.0); self.gen_p_node.setValue(0.05); self.gen_p_node.setSingleStep(0.01)
        self.gen_seed = QSpinBox(); self.gen_seed.setRange(0, 999999); self.gen_seed.setValue(42)
        self.gen_keep_shabat = QCheckBox("оставить текущие сценарии (добавить)")
        gen_form.addRow("число сценариев:", self.gen_count)
        gen_form.addRow("P(отказа канала):", self.gen_p_link)
        gen_form.addRow("P(отказа узла):", self.gen_p_node)
        gen_form.addRow("seed:", self.gen_seed)
        gen_form.addRow(self.gen_keep_shabat)
        gen_btn = QPushButton("Сгенерировать")
        gen_btn.clicked.connect(self._generate)
        gen_form.addRow(gen_btn)

        file_row = QHBoxLayout()
        for title, handler in [
            ("Загрузить JSON…", self._load_json),
            ("Сохранить JSON…", self._save_json),
            ("Загрузить CSV…", self._load_csv),
            ("Сохранить CSV…", self._save_csv),
        ]:
            b = QPushButton(title)
            b.clicked.connect(handler)
            file_row.addWidget(b)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.addWidget(editor)
        rl.addLayout(btn_row)
        rl.addWidget(gen)
        rl.addLayout(file_row)
        rl.addStretch(1)
        self.summary = QLabel("")
        rl.addWidget(self.summary)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.list)
        splitter.addWidget(right)
        splitter.setSizes([220, 480])

        lay = QVBoxLayout(self)
        lay.addWidget(splitter)
        self.setLayout(lay)

    # ------------------------------------------------------------- #

    def refresh(self) -> None:
        """Перечитать список сценариев из состояния, выбрать первый и
        обновить сводку."""
        self._loading = True
        self.list.clear()
        for s in self.state.scenarios:
            self.list.addItem(f"{s.id}  (p={s.p:.4f})")
        self._loading = False
        if self.state.scenarios:
            self.list.setCurrentRow(0)
        else:
            self._clear_editor()
        self._update_summary()

    def _clear_editor(self) -> None:
        """Очистить поля редактора сценария."""
        self.ed_id.clear()
        self.ed_p.setValue(0.0)
        self.ed_nodes.clear()
        self.ed_links.clear()

    def _on_select(self, row: int) -> None:
        """Показать выбранный в списке сценарий в полях редактора."""
        if self._loading or row < 0 or row >= len(self.state.scenarios):
            return
        s = self.state.scenarios[row]
        self.ed_id.setText(s.id)
        self.ed_p.setValue(s.p)
        self.ed_nodes.setText(_fmt_set(s.failed_nodes))
        self.ed_links.setText(_fmt_links(s.failed_links))

    def _apply_editor(self) -> None:
        """Записать поля редактора в выбранный сценарий; при ошибке разбора
        каналов — предупреждение, сценарий не меняется."""
        row = self.list.currentRow()
        if row < 0 or row >= len(self.state.scenarios):
            return
        try:
            links = _parse_links(self.ed_links.text())
        except ValueError as exc:
            QMessageBox.warning(self, "Ошибка", str(exc))
            return
        s = self.state.scenarios[row]
        s.id = self.ed_id.text().strip() or s.id
        s.p = self.ed_p.value()
        s.failed_nodes = _parse_nodes(self.ed_nodes.text())
        s.failed_links = links
        self.refresh()

    def _add_scenario(self) -> None:
        """Добавить пустой сценарий с нулевой вероятностью и выбрать его."""
        base = f"S{len(self.state.scenarios)}"
        self.state.scenarios.append(Scenario(id=base, p=0.0))
        self.refresh()
        self.list.setCurrentRow(len(self.state.scenarios) - 1)

    def _delete_selected(self) -> None:
        """Удалить выбранный в списке сценарий."""
        row = self.list.currentRow()
        if 0 <= row < len(self.state.scenarios):
            del self.state.scenarios[row]
            self.refresh()

    def _generate(self) -> None:
        """Сгенерировать случайные сценарии отказов (generate_scenarios с
        заданными p_link/p_node/seed); без флага «оставить текущие» добавляется
        штатный режим S0 (p=0.4), остальные перенормированы до суммы 0.6."""
        net = self.state.network
        if not net.nodes:
            QMessageBox.warning(self, "Нет данных", "Сначала задайте топологию сети.")
            return
        new = generate_scenarios(
            net,
            count=self.gen_count.value(),
            p_link=self.gen_p_link.value(),
            p_node=self.gen_p_node.value(),
            seed=self.gen_seed.value(),
        )
        if not self.gen_keep_shabat.isChecked():
            # перенормируем вероятности: штатный режим S0 (p=0.4) + случайные
            for s in new:
                s.p = round(0.6 / len(new), 6)
            self.state.scenarios = [Scenario(id="S0", p=0.4)] + new
        else:
            self.state.scenarios.extend(new)
        self.refresh()

    # ----------------------- файлы ----------------------- #

    def _load_json(self) -> None:
        """Загрузить сценарии из JSON-файла."""
        from PyQt6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(self, "Загрузить сценарии", "", "JSON (*.json)")
        if path:
            self.state.scenarios = load_scenarios(path)
            self.refresh()

    def _save_json(self) -> None:
        """Сохранить сценарии в JSON-файл."""
        from PyQt6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить сценарии", "scenarios.json", "JSON (*.json)")
        if path:
            save_scenarios(self.state.scenarios, path)

    def _load_csv(self) -> None:
        """Загрузить сценарии из CSV-файла."""
        from PyQt6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(self, "Загрузить сценарии (CSV)", "", "CSV (*.csv)")
        if path:
            self.state.scenarios = load_scenarios_csv(path)
            self.refresh()

    def _save_csv(self) -> None:
        """Сохранить сценарии в CSV-файл."""
        from PyQt6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить сценарии (CSV)", "scenarios.csv", "CSV (*.csv)")
        if path:
            save_scenarios_csv(self.state.scenarios, path)

    def _update_summary(self) -> None:
        """Сводка: число сценариев, Σp(s) и суммарное число отказов."""
        sc = self.state.scenarios
        total = sum(s.p for s in sc)
        n_nodes = sum(len(s.failed_nodes) for s in sc)
        n_links = sum(len(s.failed_links) for s in sc)
        self.summary.setText(
            f"Сценариев: {len(sc)}  |  Σp(s) = {total:.4f}  |  "
            f"всего отказов: узлов {n_nodes}, каналов {n_links}"
        )
