"""Вычисление показателей качества, живучести, стоимости и ограничений
согласно doc/optim.md (п. 3–4)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .model import CriticalPair, Network, Scenario, Solution


# ---------------------------------------------------------------------------
# Потоковые алгоритмы (реализация Диница — быстро для малых графов и не
# зависит от networkx, чтобы её можно было использовать в горячем цикле
# оптимизации).
# ---------------------------------------------------------------------------


class _Edge:
    """Ребро остаточной сети: конец to, индекс обратного ребра rev, остаток
    пропускной способности cap (прямое и обратное ребра хранятся парой)."""

    __slots__ = ("to", "rev", "cap")

    def __init__(self, to: int, rev: int, cap: float):
        self.to = to
        self.rev = rev
        self.cap = cap


class Dinic:
    """Максимальный поток во взвешенном ориентированном графе."""

    def __init__(self, n: int):
        """n — число вершин; g — списки смежности рёбер."""
        self.n = n
        self.g: list[list[_Edge]] = [[] for _ in range(n)]

    def add_edge(self, u: int, v: int, cap: float) -> None:
        """Направленное ребро u→v ёмкостью cap вместе с обратным ребром
        нулевой ёмкости (остатком); cap <= 0 игнорируются."""
        if cap <= 0:
            return
        self.g[u].append(_Edge(v, len(self.g[v]), cap))
        self.g[v].append(_Edge(u, len(self.g[u]) - 1, 0.0))

    def max_flow(self, s: int, t: int, limit: float = float("inf")) -> float:
        """Максимальный поток s→t, не превышающий limit (по умолчанию — без
        ограничения)."""
        flow = 0.0
        while flow < limit - 1e-12:
            level = [-1] * self.n
            level[s] = 0
            q = [s]
            for u in q:
                for e in self.g[u]:
                    if e.cap > 1e-12 and level[e.to] < 0:
                        level[e.to] = level[u] + 1
                        q.append(e.to)
            if level[t] < 0:
                return flow
            it = [0] * self.n

            def dfs(u: int, f: float) -> float:
                if u == t:
                    return f
                while it[u] < len(self.g[u]):
                    e = self.g[u][it[u]]
                    if e.cap > 1e-12 and level[e.to] == level[u] + 1:
                        d = dfs(e.to, min(f, e.cap))
                        if d > 1e-12:
                            e.cap -= d
                            self.g[e.to][e.rev].cap += d
                            return d
                    it[u] += 1
                return 0.0

            while flow < limit - 1e-12:
                f = dfs(s, min(float("inf"), limit - flow))
                if f <= 1e-12:
                    break
                flow += f
        return flow


# ---------------------------------------------------------------------------
# Вспомогательные структуры
# ---------------------------------------------------------------------------


@dataclass
class CapacityMap:
    """Остаточные ёмкости направленных пар (i, j) с учётом резервирования."""

    caps: dict[tuple[str, str], float]

    def copy(self) -> "CapacityMap":
        """Независимая копия таблицы остаточных ёмкостей."""
        return CapacityMap(dict(self.caps))


def _alive_pairs(net: Network, s: Scenario) -> set[tuple[str, str]]:
    """Пары узлов базовых каналов, оба конца которых живы в сценарии s."""
    pairs = {l.key for l in net.links}
    return {(a, b) for (a, b) in pairs if a not in s.failed_nodes and b not in s.failed_nodes}


def capacity_map(net: Network, x: Solution, s: Scenario) -> CapacityMap:
    """Ёмкости каналов в сценарии s: базовые ёмкости выживших каналов плюс
    добавленная резервная ёмкость выбранных кандидатов (доля ёмкости
    кандидата = объём резервирования)."""
    caps: dict[tuple[str, str], float] = {}
    for l in net.links:
        if l.key in s.failed_links:
            continue
        if l.source in s.failed_nodes or l.target in s.failed_nodes:
            continue
        caps[l.key] = caps.get(l.key, 0.0) + l.capacity

    vols = x.volumes()
    for k, cand in enumerate(net.candidates):
        v = vols[k]
        if v <= 0.0:
            continue
        if cand.source in s.failed_nodes or cand.target in s.failed_nodes:
            continue
        key = (cand.source, cand.target)
        caps[key] = caps.get(key, 0.0) + cand.capacity * v

    return CapacityMap(caps)


def _max_flow_value(caps: CapacityMap, idx: dict[str, int], a: str, b: str) -> float:
    """Величина максимального потока a→b по сети с остаточными ёмкостями caps
    (без изменения остатков); 0, если a или b выведены из строя."""
    if a not in idx or b not in idx:
        return 0.0
    d = Dinic(len(idx))
    for (u, v), c in caps.caps.items():
        if u in idx and v in idx:
            d.add_edge(idx[u], idx[v], c)
    return d.max_flow(idx[a], idx[b])


def _route_with_residual(
    caps: CapacityMap, idx: dict[str, int], a: str, b: str, volume: float
) -> float:
    """Максимальный поток a→b (ограниченный volume) с учётом остаточных
    ёмкостей; уменьшает остатки на величину прокачанного потока."""
    if volume <= 0.0 or a not in idx or b not in idx:
        return 0.0
    d = Dinic(len(idx))
    edges: list[tuple[tuple[str, str], _Edge]] = []
    for (u, v), c in caps.caps.items():
        # add_edge игнорирует нулевые ёмкости: фиксируем ребро только если
        # оно реально добавлено (после предыдущих потребностей остаток может
        # быть исчерпан)
        if c > 1e-12 and u in idx and v in idx:
            pos = len(d.g[idx[u]])
            d.add_edge(idx[u], idx[v], c)
            edges.append(((u, v), d.g[idx[u]][pos]))
    sent = d.max_flow(idx[a], idx[b], limit=volume)
    # остаточная ёмкость пары = cap исходного (прямого) ребра после потока
    for (u, v), e in edges:
        caps.caps[(u, v)] = e.cap
    return sent


# ---------------------------------------------------------------------------
# Показатели
# ---------------------------------------------------------------------------


def quality(
    net: Network,
    x: Solution,
    s: Scenario,
    mode: str = "flow",
) -> float:
    """Качество функционирования Q(x, s) в сценарии s.

    mode="flow" — доля доставленного трафика (последовательная маршрутизация
    потребностей через остаточные ёмкости);
    mode="connectivity" — доля приоритетных пар (из потребностей), сохраняющих
    связность хотя бы одним путём.
    """
    caps = capacity_map(net, x, s)
    idx = {nid: i for i, nid in enumerate(n for n in net.nodes if n not in s.failed_nodes)}
    total = sum(d.volume for d in net.demands)
    if total <= 0:
        return 1.0

    if mode == "connectivity":
        ok = 0
        for dem in net.demands:
            if _max_flow_value(caps, idx, dem.source, dem.target) > 1e-9:
                ok += 1
        return ok / len(net.demands) if net.demands else 1.0

    delivered = 0.0
    # потребности маршрутизуются последовательно: каждая следующая получает
    # остаток ёмкостей после предыдущих, поэтому доля доставленного ≤ 1
    for dem in net.demands:
        delivered += _route_with_residual(caps, idx, dem.source, dem.target, dem.volume)
    return min(1.0, delivered / total)


def phi(q: float, q_min: float) -> float:
    """Пороговая функция φ(Q) из п. 3 doc/optim.md."""
    return 1.0 if q >= q_min else 0.0


def survivability(
    net: Network,
    x: Solution,
    scenarios: Iterable[Scenario],
    q_min: float,
    mode: str = "flow",
) -> float:
    """Живучесть Ψ(x) = Σ p(s)·φ(Q(x, s)) (п. 3 doc/optim.md) — матожидание
    индикатора «качество не ниже Q_min» по распределению вероятностей
    сценариев поражения p(s). Здесь φ зависит только от Q; время восстановления
    T_восст учитывается отдельным критерием в свёртке (п. 6)."""
    return float(sum(s.p * phi(quality(net, x, s, mode), q_min) for s in scenarios))


def cost(net: Network, x: Solution) -> float:
    """Стоимость резервирования C_Σ(x) = Σ c_k·t_k (п. 4.2 doc/optim.md)."""
    return float(sum(c.cost * v for c, v in zip(net.candidates, x.volumes())))


def recovery_time(
    net: Network,
    x: Solution,
    scenarios: Iterable[Scenario],
    unit_time: float = 1.0,
) -> float:
    """Упрощённая модель времени восстановления: T = u·Σ p(s)·n(s)·(1−κ(s)),
    где n(s) — число отказавших каналов, κ(s) — доля отказавших каналов, для
    которых в решении x есть резерв на той же паре узлов."""
    vols = x.volumes()
    reserve_pairs = {
        (c.source, c.target) for c, v in zip(net.candidates, vols) if v > 0
    }
    acc = 0.0
    for s in scenarios:
        failed = [k for k in s.failed_links if k not in reserve_pairs]
        n_failed = len(failed) + len(s.failed_nodes)
        covered = len(s.failed_links) - len(failed)
        kappa = covered / len(s.failed_links) if s.failed_links else 1.0
        acc += s.p * n_failed * (1.0 - kappa)
    return float(unit_time * acc)


def connectivity(
    net: Network,
    x: Solution,
    s: Scenario,
    pair: CriticalPair,
) -> int:
    """Связность λ_ab(x, s) — число непересекающихся по рёбрам путей (п. 4.3)."""
    caps = capacity_map(net, x, s)
    idx = {nid: i for i, nid in enumerate(n for n in net.nodes if n not in s.failed_nodes)}
    unit = CapacityMap({k: 1.0 for k, v in caps.caps.items() if v > 1e-9})
    if pair.a not in idx or pair.b not in idx:
        return 0
    d = Dinic(len(idx))
    for (u, v) in unit.caps:
        d.add_edge(idx[u], idx[v], 1.0)
    return int(round(d.max_flow(idx[pair.a], idx[pair.b])))


# ---------------------------------------------------------------------------
# Нормировка критериев для аддитивной свёртки (п. 6 doc/optim.md)
# ---------------------------------------------------------------------------


def minmax_normalize(value: float, lo: float, hi: float) -> float:
    """Нормировка критерия в [0, 1] (для «больше = лучше»)."""
    if hi - lo < 1e-12:
        return 0.0
    return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))
