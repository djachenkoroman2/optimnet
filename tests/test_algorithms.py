"""Тесты алгоритмов: кодирование/декодирование решения, работа всех
оптимизаторов, точность ветвей и границ относительно полного перебора."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from algorithms.base import Encoding, OptimizeResult
from algorithms.bnb import BranchAndBound
from algorithms.ga import GeneticAlgorithm
from algorithms.greedy import Greedy
from algorithms.sa import SimulatedAnnealing
from algorithms.registry import OPTIMIZERS, available_optimizers, create_optimizer
from core.problem import Problem, ProblemConfig


# ---------------------------------------------------------------------------
# Кодирование / декодирование (требование к ГА из ТЗ)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["discrete", "continuous", "mixed"])
def test_encoding_roundtrip(net, scenarios, mode):
    p = Problem(net, scenarios, ProblemConfig(mode=mode))
    enc = Encoding(p)
    rng = np.random.default_rng(0)
    for _ in range(20):
        v = enc.random_vector(rng)
        v2 = enc.encode(enc.decode(v))
        assert np.all(v2 >= 0.0) and np.all(v2 <= 1.0)
        # закреплённая точка: повторное кодирование не меняет вектор
        v3 = enc.encode(enc.decode(v2))
        assert np.allclose(v2, v3)


def test_encoding_discrete_threshold(net, scenarios):
    p = Problem(net, scenarios, ProblemConfig(mode="discrete"))
    enc = Encoding(p)
    sol = enc.decode(np.array([0.4, 0.6]))
    assert sol.discrete.tolist() == [False, True]
    # дискретный режим: объёмы всегда полные
    assert np.all(sol.continuous == 1.0)


def test_encoding_continuous_all_selected(net, scenarios):
    p = Problem(net, scenarios, ProblemConfig(mode="continuous"))
    enc = Encoding(p)
    sol = enc.decode(np.array([0.3, 0.7]))
    assert sol.discrete.tolist() == [True, True]
    assert np.allclose(sol.continuous, [0.3, 0.7])


def test_encoding_mixed_positive_means_selected(net, scenarios):
    p = Problem(net, scenarios, ProblemConfig(mode="mixed"))
    enc = Encoding(p)
    sol = enc.decode(np.array([0.04, 0.5]))
    assert sol.discrete.tolist() == [False, True]
    # у выключенного кандидата фактический объём обнуляется
    assert sol.volumes()[0] == 0.0


# ---------------------------------------------------------------------------
# Работа оптимизаторов
# ---------------------------------------------------------------------------


def _check_result(res: OptimizeResult, n: int) -> None:
    assert isinstance(res, OptimizeResult)
    assert res.best.discrete.shape == (n,)
    assert res.best.continuous.shape == (n,)
    assert len(res.history) >= 1
    assert res.runtime >= 0.0
    # история лучшего значения не убывает
    assert all(b >= a - 1e-12 for a, b in zip(res.history, res.history[1:]))


def test_ga_runs_and_improves(net, scenarios):
    cfg = ProblemConfig(formulation="max_survivability", q_min=0.8, mode="discrete", budget=7.0)
    p = Problem(net, scenarios, cfg)
    ga = GeneticAlgorithm(population_size=30, generations=40)
    res = ga.optimize(p, seed=1)
    _check_result(res, p.n)
    # стартовое пустое решение имеет Ψ=0.6; ГА должен найти не хуже
    assert res.eval_result.psi >= 0.6 - 1e-9


def test_ga_callback_stop(net, scenarios):
    p = Problem(net, scenarios)
    ga = GeneticAlgorithm(population_size=10, generations=100)
    calls = []

    def cb(it, best):
        calls.append(it)
        return it < 5  # остановить после 5-й итерации

    res = ga.optimize(p, callback=cb, seed=0)
    assert res.stopped
    assert calls == [1, 2, 3, 4, 5]


def test_sa_runs(net, scenarios):
    cfg = ProblemConfig(formulation="max_survivability", q_min=0.8, mode="mixed", budget=7.0)
    p = Problem(net, scenarios, cfg)
    sa = SimulatedAnnealing(iterations=500, t0=2.0, cooling=0.99)
    res = sa.optimize(p, seed=2)
    _check_result(res, p.n)
    assert res.eval_result.psi >= 0.6 - 1e-9


def test_greedy_respects_budget(net, scenarios):
    cfg = ProblemConfig(formulation="max_survivability", q_min=0.8, mode="discrete", budget=4.0)
    p = Problem(net, scenarios, cfg)
    res = Greedy().optimize(p)
    _check_result(res, p.n)
    assert res.eval_result.cost <= 4.0 + 1e-9


def test_bnb_matches_bruteforce(net, scenarios):
    cfg = ProblemConfig(formulation="max_survivability", q_min=0.8, mode="discrete", budget=6.0)
    p = Problem(net, scenarios, cfg)
    res = BranchAndBound().optimize(p)

    best = -np.inf
    for bits in itertools.product([0, 1], repeat=p.n):
        v = np.array(bits, dtype=float)
        cost = sum(c.cost * b for c, b in zip(net.candidates, bits))
        if cost <= cfg.budget + 1e-9:
            best = max(best, p.evaluate(Encoding(p).decode(v)).psi)
    assert res.eval_result.psi == pytest.approx(best)


def test_bnb_rejects_wrong_formulation(net, scenarios):
    p = Problem(net, scenarios, ProblemConfig(formulation="min_cost", mode="discrete"))
    with pytest.raises(ValueError):
        BranchAndBound().optimize(p)


# ---------------------------------------------------------------------------
# Реестр алгоритмов (единый интерфейс для UI)
# ---------------------------------------------------------------------------


def test_registry_finds_all():
    keys = {s.key for s in OPTIMIZERS}
    assert keys == {"ga", "sa", "bnb", "greedy"}


def test_registry_filters_by_formulation():
    specs = available_optimizers("max_survivability", "discrete")
    assert {s.key for s in specs} == {"ga", "sa", "bnb", "greedy"}
    # ветви и границы недоступны в других постановках
    specs2 = available_optimizers("min_cost", "discrete")
    assert {s.key for s in specs2} == {"ga", "sa", "greedy"}


def test_registry_create_with_params():
    spec = next(s for s in OPTIMIZERS if s.key == "ga")
    ga = create_optimizer(spec, population_size=42, generations=7, unknown_param=1)
    assert ga.population_size == 42
    assert ga.generations == 7
    # значения по умолчанию из сигнатуры
    spec2 = next(s for s in OPTIMIZERS if s.key == "sa")
    sa = create_optimizer(spec2)
    assert sa.t0 == 1.0
