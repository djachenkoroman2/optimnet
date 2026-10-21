"""Тесты постановок задачи (п. 5–6 doc/optim.md): ограничения, штрафы,
целевые функции, кэширование оценок."""

from __future__ import annotations

import numpy as np
import pytest

from core.problem import Problem, ProblemConfig
from core.model import CriticalPair, Solution


def make_problem(net, scenarios, **cfg) -> Problem:
    return Problem(net, scenarios, ProblemConfig(**cfg))


def test_objective_max_survivability(net, scenarios, empty_solution):
    # q_min=0.2: Q(S0)=7/8 и Q(S1)=2/8 оба выше порога → φ=1 в обоих
    p = make_problem(net, scenarios, formulation="max_survivability", q_min=0.2)
    res = p.evaluate(empty_solution)
    assert res.psi == pytest.approx(1.0)
    assert res.objective == pytest.approx(res.psi)  # нарушений нет
    assert res.feasible


def test_budget_violation(net, scenarios, full_solution):
    # полное резервирование стоит 3+4=7 > бюджета 5
    p = make_problem(net, scenarios, formulation="max_survivability", budget=5.0, q_min=0.8)
    res = p.evaluate(full_solution)
    assert not res.feasible
    assert res.violations["budget"] == pytest.approx(2.0)
    # штраф уменьшает целевую функцию
    assert res.objective < res.psi


def test_quality_violation(net, scenarios, empty_solution):
    # Q_min выше любого достижимого качества → нарушение по всем сценариям
    p = make_problem(net, scenarios, q_min=0.99)
    res = p.evaluate(empty_solution)
    assert "quality" in res.violations
    assert not res.feasible


def test_min_cost_formulation(net, scenarios, full_solution):
    # q_min=0.2: полное резервирование не нарушает ограничений по качеству
    p = make_problem(net, scenarios, formulation="min_cost", budget=10.0, q_min=0.2)
    res = p.evaluate(full_solution)
    assert res.feasible
    assert res.objective == pytest.approx(-res.cost)
    # Ψ_треб=1.1 недостижимо → штраф за psi_required
    p2 = make_problem(net, scenarios, formulation="min_cost", psi_required=1.1, budget=10.0, q_min=0.2)
    res2 = p2.evaluate(full_solution)
    assert "psi_required" in res2.violations


def test_connectivity_constraint(net, scenarios, empty_solution):
    # требуется λ_AC ≥ 2 во всех сценариях; в S1 (отказ B) прямой путь есть,
    # но путь через B отсутствует — однако A,C живы, λ по выжившим рёбрам = 1
    p = Problem(net, scenarios, ProblemConfig(q_min=0.8), critical_pairs=[CriticalPair("A", "C", r=2)])
    res = p.evaluate(empty_solution)
    assert "connectivity" in res.violations


def test_minimax_formulation(net, scenarios, empty_solution, full_solution):
    p = make_problem(net, scenarios, formulation="minimax", q_min=0.0)
    r_empty = p.evaluate(empty_solution)
    r_full = p.evaluate(full_solution)
    # минимакс: цель — худший Q по сценариям; отказ узла B не перекрывается
    # резервом (кандидаты инцидентны B), поэтому худший Q = 2/8 у обоих
    assert r_empty.objective == pytest.approx(2.0 / 8.0)
    assert r_full.objective == pytest.approx(2.0 / 8.0)
    # а при отказе только канала (A,B) резерв C1 (ёмкость 10) полностью
    # перекрывает отказ: Q(S1) = 1.0
    s_link = [scenarios[0], scenarios[1].__class__("S1", p=0.4, failed_links={("A", "B")})]
    p2 = Problem(net, s_link, ProblemConfig(formulation="minimax", q_min=0.0))
    assert p2.evaluate(full_solution).objective == pytest.approx(1.0)
    assert p2.evaluate(empty_solution).objective == pytest.approx(2.0 / 8.0)


def test_scalarized_formulation(net, scenarios, empty_solution):
    cfg = ProblemConfig(formulation="scalarized", weights=(1.0, 0.0, 0.0, 0.0), q_min=0.2)
    p = Problem(net, scenarios, cfg)
    res = p.evaluate(empty_solution)
    # все веса на Ψ — свёртка совпадает с Ψ
    assert res.objective == pytest.approx(res.psi)


def test_weights_sum_and_normalization(net, scenarios, full_solution):
    cfg = ProblemConfig(formulation="scalarized", weights=(0.25, 0.25, 0.25, 0.25), budget=10.0, q_min=0.2)
    p = Problem(net, scenarios, cfg)
    res = p.evaluate(full_solution)
    assert res.feasible
    # нормированные слагаемые в [0,1] → свёртка тоже
    assert 0.0 <= res.objective <= 1.0 + 1e-9


def test_evaluation_cache(net, scenarios, empty_solution):
    p = make_problem(net, scenarios)
    p.evaluate(empty_solution)
    p.evaluate(empty_solution)
    assert p.evaluations == 1
    other = Solution(np.array([True, False]), np.array([1.0, 1.0]))
    p.evaluate(other)
    assert p.evaluations == 2


def test_discrete_mode_forces_full_volume(net, scenarios):
    cfg = ProblemConfig(mode="discrete")
    p = Problem(net, scenarios, cfg)
    # объёмы игнорируются: включённый кандидат даёт полную ёмкость
    x = Solution(np.array([True, False]), np.array([0.1, 0.1]))
    res = p.evaluate(x)
    vol_solution = Solution(np.array([True, False]), np.array([1.0, 0.1]))
    res2 = p.evaluate(vol_solution)
    assert res.psi == res2.psi
