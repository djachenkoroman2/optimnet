"""Вкладка «Топология»: редактируемые таблицы узлов, каналов, потребностей
и кандидатов на резервирование + визуализация графа."""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView, QHBoxLayout, QHeaderView, QLabel, QPushButton,
    QSplitter, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from core.model import Candidate, Demand, Link, Network, Node

from .graph_canvas import NetworkCanvas
from .state import AppState

_KINDS_RU = {"router": "маршрутизатор", "switch": "коммутатор", "server": "сервер"}
_RU_KINDS = {v: k for k, v in _KINDS_RU.items()}


def _num(text: str, default: float = 0.0) -> float:
    """Разобрать число из ячейки таблицы (запятая допустима); при ошибке —
    default."""
    try:
        return float(text.replace(",", "."))
    except ValueError:
        return default


class _Table(QTableWidget):
    """Таблица с однотипными числовыми/текстовыми ячейками."""

    def __init__(self, headers: list[str], parent=None):
        """Пустая таблица с заголовками headers и выделением строк."""
        super().__init__(0, len(headers), parent)
        self.setHorizontalHeaderLabels(headers)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.verticalHeader().setVisible(False)

    def add_row(self, values: list[object]) -> None:
        """Добавить строку из значений values (приводятся к строкам)."""
        r = self.rowCount()
        self.insertRow(r)
        for c, v in enumerate(values):
            self.setItem(r, c, QTableWidgetItem(str(v)))

    def rows(self) -> list[list[str]]:
        """Все строки таблицы как список списков строк (пустая ячейка — '')."""
        out = []
        for r in range(self.rowCount()):
            out.append([
                self.item(r, c).text().strip() if self.item(r, c) else ""
                for c in range(self.columnCount())
            ])
        return out

    def remove_selected(self) -> None:
        """Удалить выбранные строки (снизу вверх, чтобы индексы не съезжали)."""
        rows = sorted({i.row() for i in self.selectedIndexes()}, reverse=True)
        for r in rows:
            self.removeRow(r)


class TopologyTab(QWidget):
    """Вкладка «Топология»: редактируемые таблицы узлов, каналов, потребностей
    и кандидатов на резервирование + визуализация графа (NetworkCanvas)."""

    def __init__(self, state: AppState, parent=None):
        """Собрать таблицы (во внутренних вкладках), холст графа и кнопку
        применения таблиц к модели."""
        super().__init__(parent)
        self.state = state

        self.nodes_table = _Table(["id", "тип"])
        self.links_table = _Table(["источник", "назначение", "ёмкость", "стоимость", "задержка", "надёжность"])
        self.demands_table = _Table(["источник", "назначение", "объём"])
        self.candidates_table = _Table(["id", "источник", "назначение", "ёмкость", "стоимость"])

        tables = QTabWidget()
        for title, table, hint in [
            ("Узлы", self.nodes_table, "Тип: маршрутизатор / коммутатор / сервер"),
            ("Каналы", self.links_table, "Ориентированные каналы базовой топологии"),
            ("Потребности", self.demands_table, "Пары узлов с трафиком (Q = доля доставленного)"),
            ("Кандидаты", self.candidates_table, "Возможные резервные каналы (переменные x)"),
        ]:
            w = QWidget()
            lay = QVBoxLayout(w)
            btn_row = QHBoxLayout()
            add_btn = QPushButton("Добавить строку")
            del_btn = QPushButton("Удалить выбранные")
            add_btn.clicked.connect(lambda _=False, t=table: t.add_row([""] * t.columnCount()))
            del_btn.clicked.connect(table.remove_selected)
            btn_row.addWidget(add_btn)
            btn_row.addWidget(del_btn)
            btn_row.addWidget(QLabel(hint))
            btn_row.addStretch(1)
            lay.addLayout(btn_row)
            lay.addWidget(table)
            tables.addTab(w, title)

        self.canvas = NetworkCanvas()

        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.addWidget(self.canvas)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(tables)
        splitter.addWidget(right)
        splitter.setSizes([520, 480])

        apply_btn = QPushButton("Применить таблицы → модель")
        apply_btn.clicked.connect(self.apply_tables)

        lay = QVBoxLayout(self)
        lay.addWidget(splitter)
        lay.addWidget(apply_btn)
        self.setLayout(lay)

    # ------------------------------------------------------------- #

    def refresh(self) -> None:
        """Заполнить таблицы из state.network (типы узлов — по-русски) и
        перерисовать граф."""
        net = self.state.network
        self.nodes_table.setRowCount(0)
        for n in net.nodes.values():
            self.nodes_table.add_row([n.id, _KINDS_RU.get(n.kind, n.kind)])
        self.links_table.setRowCount(0)
        for l in net.links:
            self.links_table.add_row([l.source, l.target, l.capacity, l.cost, l.delay, l.reliability])
        self.demands_table.setRowCount(0)
        for d in net.demands:
            self.demands_table.add_row([d.source, d.target, d.volume])
        self.candidates_table.setRowCount(0)
        for c in net.candidates:
            self.candidates_table.add_row([c.id, c.source, c.target, c.capacity, c.cost])
        self.redraw_graph()

    def redraw_graph(self) -> None:
        """Перерисовать граф текущей сети (раскладку узлов считать заново)."""
        self.canvas.clear_positions()
        self.canvas.draw_network(self.state.network)

    def apply_tables(self) -> None:
        """Собрать модель Network из таблиц (типы узлов — обратно в EN),
        проверить ссылочную целостность; при ошибке — предупреждение,
        состояние не меняется. Успех — обновляет state и сбрасывает результаты."""
        net = Network()
        for rid, kind in self.nodes_table.rows():
            if not rid:
                continue
            net.nodes[rid] = Node(id=rid, kind=_RU_KINDS.get(kind, kind or "router"))
        for s, t, cap, cost, delay, rel in self.links_table.rows():
            if s and t:
                net.links.append(Link(s, t, _num(cap), _num(cost, 1.0), _num(delay), _num(rel, 0.99)))
        for s, t, vol in self.demands_table.rows():
            if s and t:
                net.demands.append(Demand(s, t, _num(vol, 1.0)))
        for cid, s, t, cap, cost in self.candidates_table.rows():
            if cid:
                net.candidates.append(Candidate(cid, s, t, _num(cap), _num(cost, 1.0)))
        try:
            net.validate()
        except ValueError as exc:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Ошибка топологии", str(exc))
            return
        self.state.network = net
        self.state.last_results = {}
        self.redraw_graph()
