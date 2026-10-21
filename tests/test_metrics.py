"""Тесты расчёта показателей Q(x, s), Ψ(x), C_Σ, T_восст и связности λ_ab
согласно п. 3–4 doc/optim.md."""

from __future__ import annotations

import numpy as np
import pytest

from core import metrics
from core.model import CriticalPair, Scenario, Solution


def test_flow_quality_without_reserve(net, empty_solution):
    # макс. поток A→C = прямой 2 + через B 5 = 7, доля = 7/8
    q = metrics.quality(net, empty_solution, Scenario("S", p=1.0))
    assert q == _ap(7.0 / 8.0)


def test_flow_quality_with_full_reserve(net, full_solution):
    # 2 + (5+10) = 17 ≥ 8 → весь трафик доставлен
    q = metrics.quality(net, full_solution, Scenario("S", p=1.0))
    assert q == _ap(1.0)


def test_flow_quality_failed_node(net, empty_solution):
    # отказ B: остаётся только прямой канал A→C (ёмкость 2)
    s = Scenario("S", p=1.0, failed_nodes={"B"})
    q = metrics.quality(net, empty_solution, s)
    assert q == _ap(2.0 / 8.0)


def test_flow_quality_failed_link(net, empty_solution):
    # отказ канала A→C: остаётся путь A→B→C (ёмкость 5)
    s = Scenario("S", p=1.0, failed_links={("A", "C")})
    q = metrics.quality(net, empty_solution, s)
    assert q == _ap(5.0 / 8.0)


def test_failed_node_kills_incident_links(net, full_solution):
    # отказ B обрубает и резерв C1/C2 (они инцидентны B): остаётся A→C = 2
    s = Scenario("S", p=1.0, failed_nodes={"B"})
    q = metrics.quality(net, full_solution, s)
    assert q == _ap(2.0 / 8.0)


def test_connectivity_quality_mode(net, empty_solution):
    net2 = net
    net2.demands = [
        net2.demands[0],
        net2.demands[0].__class__("B", "C", volume=1.0),
    ]
    s = Scenario("S", p=1.0, failed_links={("B", "C")})
    q = metrics.quality(net2, empty_solution, s, mode="connectivity")
    # A→C связно, B→C — нет
    assert q == _ap(0.5)


def test_phi_threshold():
    assert metrics.phi(0.8, 0.8) == 1.0
    assert metrics.phi(0.79, 0.8) == 0.0


def test_survivability(net, scenarios, empty_solution, full_solution):
    # пустое решение: Q(S0)=7/8 ≥ 0.8 → φ=1; Q(S1)=2/8 < 0.8 → φ=0
    psi_empty = metrics.survivability(net, empty_solution, scenarios, q_min=0.8)
    assert psi_empty == _ap(0.6)
    # полное резервирование не спасает от отказа узла B: кандидаты C1/C2
    # инцидентны B и в сценарии S1 тоже отказывают → Ψ прежняя
    psi_full = metrics.survivability(net, full_solution, scenarios, q_min=0.8)
    assert psi_full == _ap(0.6)
    # зато при отказе только канала (A,B) резерв C1 восстанавливает Q=7/8 ≥ Q_min
    s_link = [Scenario("S0", p=0.6), Scenario("S1", p=0.4, failed_links={("A", "B")})]
    assert metrics.survivability(net, full_solution, s_link, q_min=0.8) == _ap(1.0)


def test_cost(net):
    x = Solution(discrete=np.array([True, True]), continuous=np.array([1.0, 0.5]))
    # C1 полностью: 3·1 + C2 наполовину: 4·0.5 = 5
    assert metrics.cost(net, x) == _ap(5.0)
    # выключенный кандидат не стоит ничего, даже если объём ненулевой
    x2 = Solution(discrete=np.array([False, True]), continuous=np.array([1.0, 1.0]))
    assert metrics.cost(net, x2) == _ap(4.0)


def test_recovery_time(net, scenarios):
    # S1: отказ узла B и канала (A,B); без резерва κ=0, n=2, p=0.4
    s1 = Scenario("S1", p=0.4, failed_nodes={"B"}, failed_links={("A", "B")})
    t0 = metrics.recovery_time(net, Solution.empty(2), [s1], unit_time=2.0)
    assert t0 == _ap(2.0 * 0.4 * 2.0)
    # резерв на паре (A,B) покрывает отказ канала: κ=1, остаётся только узел B
    x = Solution(discrete=np.array([True, False]), continuous=np.array([1.0, 0.0]))
    t1 = metrics.recovery_time(net, x, [s1], unit_time=2.0)
    assert t1 == _ap(0.0)


def test_connectivity_edge_disjoint_paths(net, empty_solution):
    # пути A→C (прямой) и A→B→C непересекаются по рёбрам → λ=2
    lam = metrics.connectivity(net, empty_solution, Scenario("S", p=1.0), CriticalPair("A", "C", r=2))
    assert lam == 2
    # отказ прямого канала оставляет один путь
    s = Scenario("S", p=1.0, failed_links={("A", "C")})
    assert metrics.connectivity(net, empty_solution, s, CriticalPair("A", "C", r=2)) == 1


def test_minmax_normalize():
    assert metrics.minmax_normalize(5.0, 0.0, 10.0) == _ap(0.5)
    assert metrics.minmax_normalize(15.0, 0.0, 10.0) == _ap(1.0)
    assert metrics.minmax_normalize(-5.0, 0.0, 10.0) == _ap(0.0)
    # вырожденный интервал — ноль, без деления на ноль
    assert metrics.minmax_normalize(3.0, 3.0, 3.0) == 0.0


def _ap(v: float):

    return pytest.approx(v, abs=1e-9)
