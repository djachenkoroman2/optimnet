"""Генерация демонстрационных примеров (examples/*.json).

Топология — кольцо маршрутизаторов с присоединёнными серверами; кандидаты на
резервирование — дополнительные хорды/каналы. Сценарии: штатный режим S0
(без отказов) + случайные отказы с фиксированным seed.

Запуск:  python -m examples.generate_examples
"""

from __future__ import annotations

import json
from pathlib import Path

from core.io import generate_scenarios
from core.model import Candidate, Demand, Link, Network, Node, Scenario

HERE = Path(__file__).parent


def _ring_network(
    n_routers: int,
    server_pairs: list[tuple[str, int, int]],
    chords: list[tuple[int, int]],
    capacity: float = 10.0,
    chord_capacity: float = 10.0,
) -> Network:
    """Построить сеть-кольцо: маршрутизаторы R1..Rn с двунаправленными
    каналами, серверы srv, присоединённые к маршрутизаторам a и b, полный
    набор потребностей между серверами и двунаправленные пары кандидатов-
    хорд (переменные x)."""
    nodes: dict[str, Node] = {}
    links: list[Link] = []
    for i in range(1, n_routers + 1):
        nodes[f"R{i}"] = Node(id=f"R{i}", kind="router")
    for srv, a, b in server_pairs:
        nodes[srv] = Node(id=srv, kind="server")
    # кольцо: двунаправленные каналы
    for i in range(1, n_routers + 1):
        j = i % n_routers + 1
        links.append(Link(f"R{i}", f"R{j}", capacity=capacity, cost=2.0))
        links.append(Link(f"R{j}", f"R{i}", capacity=capacity, cost=2.0))
    # серверы к маршрутизаторам
    for srv, a, b in server_pairs:
        links.append(Link(srv, f"R{a}", capacity=capacity, cost=1.0))
        links.append(Link(f"R{a}", srv, capacity=capacity, cost=1.0))
        links.append(Link(srv, f"R{b}", capacity=capacity, cost=1.0))
        links.append(Link(f"R{b}", srv, capacity=capacity, cost=1.0))
    demands = [
        Demand(srv_a, srv_b, volume=8.0)
        for srv_a, _, _ in server_pairs
        for srv_b, _, _ in server_pairs
        if srv_a < srv_b
    ]
    candidates = [
        Candidate(
            id=f"C{k+1}",
            source=f"R{a}",
            target=f"R{b}",
            capacity=chord_capacity,
            cost=round(4.0 + 2.0 * k, 1),
        )
        for k, (a, b) in enumerate(chords)
    ]
    # хорды — двунаправленные пары кандидатов
    for k, (a, b) in enumerate(chords):
        candidates.append(
            Candidate(
                id=f"C{k+1}r",
                source=f"R{b}",
                target=f"R{a}",
                capacity=chord_capacity,
                cost=round(4.0 + 2.0 * k, 1),
            )
        )
    net = Network(nodes=nodes, links=links, demands=demands, candidates=candidates)
    net.validate()
    return net


def _scenarios(net: Network, count: int, seed: int, p0: float = 0.4) -> list[Scenario]:
    """Сценарии для примера: штатный режим S0 с вероятностью p0 плюс count
    случайных отказов (generate_scenarios), перенормированных до 1 − p0."""
    random_part = generate_scenarios(net, count=count, seed=seed)
    for s in random_part:
        s.p = round((1.0 - p0) / count, 6)
    return [Scenario(id="S0", p=p0)] + random_part


def build_small() -> tuple[Network, list[Scenario]]:
    """Малый пример: 4 маршрутизатора, 2 сервера, 2 хорды (examples/small.json)."""
    net = _ring_network(
        n_routers=4,
        server_pairs=[("S1", 1, 2), ("S2", 3, 4)],
        chords=[(1, 3), (2, 4)],
    )
    return net, _scenarios(net, count=6, seed=7)


def build_medium() -> tuple[Network, list[Scenario]]:
    """Средний пример: 8 маршрутизаторов, 2 сервера, 3 хорды
    (examples/medium.json — грузится в GUI по умолчанию)."""
    net = _ring_network(
        n_routers=8,
        server_pairs=[("S1", 1, 2), ("S2", 4, 5)],
        chords=[(1, 4), (2, 6), (3, 7)],
    )
    return net, _scenarios(net, count=8, seed=21)


def build_large() -> tuple[Network, list[Scenario]]:
    """Большой пример: 16 маршрутизаторов, 4 сервера, 6 хорд
    (examples/large.json)."""
    net = _ring_network(
        n_routers=16,
        server_pairs=[("S1", 1, 2), ("S2", 8, 9), ("S3", 4, 12), ("S4", 6, 14)],
        chords=[(1, 8), (2, 9), (3, 10), (4, 12), (5, 13), (6, 14)],
    )
    return net, _scenarios(net, count=10, seed=42)


def main() -> None:
    """Сгенерировать все три примера и перезаписать examples/*.json."""
    for builder, fname in [(build_small, "small.json"), (build_medium, "medium.json"), (build_large, "large.json")]:
        net, scenarios = builder()
        data = net.to_dict()
        data["scenarios"] = [s.to_dict() for s in scenarios]
        path = HERE / fname
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"{path}: узлов={len(net.nodes)}, каналов={len(net.links)}, "
              f"кандидатов={len(net.candidates)}, сценариев={len(scenarios)}")


if __name__ == "__main__":
    main()
