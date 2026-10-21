"""Визуализация графа сети: базовые каналы, отказы выбранного сценария
(красный) и резервные каналы найденного решения (зелёный)."""

from __future__ import annotations

import matplotlib

matplotlib.use("QtAgg")
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import networkx as nx

from core.model import Network, Scenario, Solution

NODE_COLORS = {"router": "#1f77b4", "switch": "#ff7f0e", "server": "#2ca02c"}


class NetworkCanvas(FigureCanvas):
    """Холст matplotlib (бэкенд QtAgg) для отрисовки графа сети.

    Показывает узлы (цвет по типу), базовые каналы, отказы выбранного
    сценария и резервные каналы найденного решения; координаты узлов
    раскладки кэшируются в _pos, чтобы граф не «прыгал» между перерисовками.
    """

    def __init__(self, parent=None):
        """Пустой холст с одной осью; _pos=None — раскладка ещё не считана."""
        self.fig = Figure(figsize=(6, 5), tight_layout=True)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111)
        self._pos: dict | None = None

    def draw_network(
        self,
        net: Network,
        scenario: Scenario | None = None,
        solution: Solution | None = None,
        title: str = "",
    ) -> None:
        """Отрисовать сеть net; scenario — подсветить отказы, solution —
        показать резервные каналы с объёмами, title — заголовок оси."""
        self.ax.clear()
        if not net.nodes:
            self.ax.text(0.5, 0.5, "Нет данных: загрузите пример\nили задайте топологию",
                         ha="center", va="center", transform=self.ax.transAxes, color="gray")
            self.draw_idle()
            return

        g = nx.DiGraph()
        for nid, node in net.nodes.items():
            g.add_node(nid, kind=node.kind)
        for l in net.links:
            g.add_edge(l.source, l.target, capacity=l.capacity)

        if self._pos is None or set(self._pos) != set(net.nodes):
            # координатная сетка узлов: силовая раскладка с фиксированным seed
            # (детерминированно для данного графа), результат кэшируется
            self._pos = nx.spring_layout(g, seed=42, k=1.2)

        pos = self._pos
        failed_nodes = scenario.failed_nodes if scenario else set()
        failed_links = scenario.failed_links if scenario else set()

        # резерв решения: пары узлов с положительным объёмом
        reserve_pairs: dict[tuple[str, str], float] = {}
        if solution is not None:
            vols = solution.volumes()
            for cand, v in zip(net.candidates, vols):
                if v > 0:
                    reserve_pairs[(cand.source, cand.target)] = v

        # базовые каналы
        base_edges = [e for e in g.edges if e not in failed_links and e not in reserve_pairs]
        nx.draw_networkx_edges(g, pos, ax=self.ax, edgelist=base_edges, edge_color="#999999",
                               arrows=True, arrowsize=8, width=1.2, alpha=0.8)
        # отказавшие каналы — красные пунктирные
        dead_edges = [e for e in g.edges if e in failed_links]
        if dead_edges:
            nx.draw_networkx_edges(g, pos, ax=self.ax, edgelist=dead_edges, edge_color="#d62728",
                                   arrows=True, arrowsize=8, width=1.6, style="dashed", alpha=0.9)
        # резервные каналы решения — зелёные жирные (пунктир если пара частично
        # совпадает с отказавшей — такое бывает при совпадении пар)
        for (u, v), volume in reserve_pairs.items():
            color = "#2ca02c" if (u, v) not in failed_links else "#d62728"
            style = "solid" if (u, v) not in failed_links else "dashed"
            g.add_edge(u, v)
            nx.draw_networkx_edges(g, pos, ax=self.ax, edgelist=[(u, v)], edge_color=color,
                                   arrows=True, arrowsize=10, width=1.5 + 2.5 * min(volume, 1.0),
                                   style=style, alpha=0.95)
            # подпись объёма резервирования у середины канала
            self.ax.text((pos[u][0] + pos[v][0]) / 2, (pos[u][1] + pos[v][1]) / 2,
                         f"{volume:.2f}", fontsize=7, color="#2ca02c", ha="center")

        # узлы: цвет по типу, отказавшие — красная кайма
        by_kind: dict[str, list[str]] = {}
        for nid in g.nodes:
            kind = net.nodes[nid].kind if nid in net.nodes else "router"
            by_kind.setdefault(kind, []).append(nid)
        for kind, nids in by_kind.items():
            nx.draw_networkx_nodes(g, pos, ax=self.ax, nodelist=nids,
                                   node_color=NODE_COLORS.get(kind, "#1f77b4"),
                                   node_size=500, alpha=0.9)
        if failed_nodes:
            nx.draw_networkx_nodes(g, pos, ax=self.ax, nodelist=list(failed_nodes),
                                   node_color="none", node_size=500,
                                   linewidths=2.5, edgecolors="#d62728")
        nx.draw_networkx_labels(g, pos, ax=self.ax, font_size=8, font_color="white")

        # легенда
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], color="#999999", lw=1.2, label="базовый канал"),
            Line2D([0], [0], color="#d62728", lw=1.6, ls="--", label="отказ (сценарий)"),
            Line2D([0], [0], color="#2ca02c", lw=3, label="резерв (решение)"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#d62728",
                   markeredgecolor="#d62728", markersize=8, label="отказ узла"),
        ]
        for kind, color in NODE_COLORS.items():
            handles.append(Line2D([0], [0], marker="o", color="w", markerfacecolor=color,
                                  markersize=8, label=kind))
        self.ax.legend(handles=handles, loc="upper left", fontsize=7, framealpha=0.9)
        self.ax.set_title(title or "Топология сети", fontsize=10)
        self.ax.axis("off")  # координатные оси скрыты — остаётся только граф
        self.draw_idle()

    def clear_positions(self) -> None:
        """Сбросить кэш раскладки: следующая отрисовка посчитает координаты
        заново (нужно при изменении топологии)."""
        self._pos = None
