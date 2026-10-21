"""Общие фикстуры: малая сеть с заранее вычислимыми показателями.

Топология:  A ──5──▶ B ──5──▶ C,  прямой канал A ──2──▶ C.
Потребность A→C объёмом 8: максимальный поток без резерва 2+5=7,
значит Q = 7/8 = 0.875. Кандидаты резервирования: C1 (A→B, 10, цена 3),
C2 (B→C, 10, цена 4).
"""

from __future__ import annotations

import numpy as np
import pytest

from core.model import Candidate, Demand, Link, Network, Node, Scenario, Solution


@pytest.fixture
def net() -> Network:
    return Network(
        nodes={
            "A": Node("A", "server"),
            "B": Node("B", "router"),
            "C": Node("C", "server"),
        },
        links=[
            Link("A", "B", capacity=5.0, cost=2.0),
            Link("B", "C", capacity=5.0, cost=2.0),
            Link("A", "C", capacity=2.0, cost=1.0),
        ],
        demands=[Demand("A", "C", volume=8.0)],
        candidates=[
            Candidate("C1", "A", "B", capacity=10.0, cost=3.0),
            Candidate("C2", "B", "C", capacity=10.0, cost=4.0),
        ],
    )


@pytest.fixture
def scenarios() -> list[Scenario]:
    """S0 (p=0.6) — штатный режим, S1 (p=0.4) — отказ узла B."""
    return [
        Scenario("S0", p=0.6),
        Scenario("S1", p=0.4, failed_nodes={"B"}),
    ]


@pytest.fixture
def empty_solution(net) -> Solution:
    return Solution.empty(len(net.candidates))


@pytest.fixture
def full_solution(net) -> Solution:
    return Solution(
        discrete=np.ones(len(net.candidates), dtype=bool),
        continuous=np.ones(len(net.candidates)),
    )
