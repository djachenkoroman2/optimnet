"""Формальная модель сети согласно doc/optim.md (п. 2).

Сеть — взвешенный ориентированный граф G = <V, E, C, W>:
узлы V (маршрутизаторы/коммутаторы/серверы), каналы E с ёмкостями C
и весовыми характеристиками W (стоимость, задержка, надёжность).

Решение x — смешанный дискретно-непрерывный вектор над пулом кандидатов
на резервирование: дискретная часть задаёт выбранные резервные каналы,
непрерывная — объём резервирования (доля ёмкости).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

NODE_KINDS = ("router", "switch", "server")


@dataclass
class Node:
    """Узел сети: идентификатор и тип (router | switch | server)."""

    id: str
    kind: str = "router"

    def to_dict(self) -> dict[str, Any]:
        """Сериализация в JSON-совместимый словарь."""
        return {"id": self.id, "kind": self.kind}

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Node":
        """Обратное преобразование to_dict; kind по умолчанию — «router»."""
        return Node(id=str(d["id"]), kind=str(d.get("kind", "router")))


@dataclass
class Link:
    """Канал связи (i, j)."""

    source: str
    target: str
    capacity: float
    cost: float = 1.0
    delay: float = 0.0
    reliability: float = 0.99

    def to_dict(self) -> dict[str, Any]:
        """Сериализация в JSON-совместимый словарь."""
        return {
            "source": self.source,
            "target": self.target,
            "capacity": self.capacity,
            "cost": self.cost,
            "delay": self.delay,
            "reliability": self.reliability,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Link":
        """Обратное преобразование to_dict с значениями по умолчанию."""
        return Link(
            source=str(d["source"]),
            target=str(d["target"]),
            capacity=float(d["capacity"]),
            cost=float(d.get("cost", 1.0)),
            delay=float(d.get("delay", 0.0)),
            reliability=float(d.get("reliability", 0.99)),
        )

    @property
    def key(self) -> tuple[str, str]:
        """Ключ пары (source, target) — для множеств отказавших каналов."""
        return (self.source, self.target)


@dataclass
class Demand:
    """Потребность в трафике между парой узлов (доля доставленного трафика)."""

    source: str
    target: str
    volume: float

    def to_dict(self) -> dict[str, Any]:
        """Сериализация в JSON-совместимый словарь."""
        return {"source": self.source, "target": self.target, "volume": self.volume}

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Demand":
        """Обратное преобразование to_dict."""
        return Demand(str(d["source"]), str(d["target"]), float(d["volume"]))


@dataclass
class Candidate:
    """Кандидат на резервирование: дополнительный канал (u, v)."""

    id: str
    source: str
    target: str
    capacity: float
    cost: float

    def to_dict(self) -> dict[str, Any]:
        """Сериализация в JSON-совместимый словарь."""
        return {
            "id": self.id,
            "source": self.source,
            "target": self.target,
            "capacity": self.capacity,
            "cost": self.cost,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Candidate":
        """Обратное преобразование to_dict."""
        return Candidate(
            id=str(d["id"]),
            source=str(d["source"]),
            target=str(d["target"]),
            capacity=float(d["capacity"]),
            cost=float(d["cost"]),
        )


@dataclass
class Scenario:
    """Сценарий поражения/отказа s с вероятностью p(s) (doc/optim.md, п. 2)."""

    id: str
    p: float
    failed_nodes: set[str] = field(default_factory=set)
    failed_links: set[tuple[str, str]] = field(default_factory=set)

    def to_dict(self) -> dict[str, Any]:
        """Сериализация; множества приводятся к сортированным спискам."""
        return {
            "id": self.id,
            "p": self.p,
            "failed_nodes": sorted(self.failed_nodes),
            "failed_links": [list(k) for k in sorted(self.failed_links)],
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Scenario":
        """Обратное преобразование to_dict; множества собираются из списков."""
        return Scenario(
            id=str(d["id"]),
            p=float(d["p"]),
            failed_nodes={str(x) for x in d.get("failed_nodes", [])},
            failed_links={(str(a), str(b)) for a, b in d.get("failed_links", [])},
        )


@dataclass
class Network:
    """Граф G = <V, E, C, W> вместе с потребностями и пулом кандидатов."""

    nodes: dict[str, Node] = field(default_factory=dict)
    links: list[Link] = field(default_factory=list)
    demands: list[Demand] = field(default_factory=list)
    candidates: list[Candidate] = field(default_factory=list)

    # ---- сериализация ----

    def to_dict(self) -> dict[str, Any]:
        """Сериализация всей сети (узлы/каналы/потребности/кандидаты)."""
        return {
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "links": [l.to_dict() for l in self.links],
            "demands": [d.to_dict() for d in self.demands],
            "candidates": [c.to_dict() for c in self.candidates],
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Network":
        """Загрузить сеть из словаря to_dict с проверкой ссылочной целостности."""
        nodes = {n.id: n for n in (Node.from_dict(x) for x in d.get("nodes", []))}
        net = Network(
            nodes=nodes,
            links=[Link.from_dict(x) for x in d.get("links", [])],
            demands=[Demand.from_dict(x) for x in d.get("demands", [])],
            candidates=[Candidate.from_dict(x) for x in d.get("candidates", [])],
        )
        net.validate()
        return net

    def validate(self) -> None:
        """Проверить, что все концы каналов, потребностей и кандидатов —
        известные узлы; иначе ValueError."""
        for l in self.links:
            if l.source not in self.nodes or l.target not in self.nodes:
                raise ValueError(f"Канал {l.key}: неизвестный узел")
        for d in self.demands:
            if d.source not in self.nodes or d.target not in self.nodes:
                raise ValueError(f"Потребность {(d.source, d.target)}: неизвестный узел")
        for c in self.candidates:
            if c.source not in self.nodes or c.target not in self.nodes:
                raise ValueError(f"Кандидат {c.id}: неизвестный узел")

    # ---- индексация ----

    def node_index(self) -> dict[str, int]:
        """Соответствие id узла → индекс (для алгоритмов на матрицах/потоках)."""
        return {nid: i for i, nid in enumerate(self.nodes)}

    def link_capacity(self) -> dict[tuple[str, str], float]:
        """Ёмкости базовых каналов по парам (source, target)."""
        return {l.key: l.capacity for l in self.links}


@dataclass
class Solution:
    """Вектор решения x: дискретная часть (выбранные кандидаты) и непрерывная
    (объёмы резервирования в [0, 1])."""

    discrete: np.ndarray
    continuous: np.ndarray

    @staticmethod
    def empty(n: int) -> "Solution":
        """Пустое решение (ничего не резервируем) размерности n."""
        return Solution(
            discrete=np.zeros(n, dtype=bool),
            continuous=np.zeros(n, dtype=float),
        )

    def volumes(self) -> np.ndarray:
        """Фактически добавленная доля ёмкости по каждому кандидату."""
        return np.where(self.discrete, self.continuous, 0.0)

    def copy(self) -> "Solution":
        """Независимая копия (копии массивов)."""
        return Solution(self.discrete.copy(), self.continuous.copy())


@dataclass
class CriticalPair:
    """Приоритетная пара узлов (a, b) с требуемым уровнем резервирования r_ab."""

    a: str
    b: str
    r: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Сериализация в JSON-совместимый словарь."""
        return {"a": self.a, "b": self.b, "r": self.r}

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "CriticalPair":
        """Обратное преобразование to_dict; r по умолчанию — 1."""
        return CriticalPair(str(d["a"]), str(d["b"]), int(d.get("r", 1)))
