"""Генетический алгоритм (глава 3 doc/optim.md).

Кодирование хромосомы — вещественный вектор [0, 1]^n (см. Encoding):
дискретная часть — выбранные резервные каналы, непрерывная — объёмы.
Селекция турнирная, кроссинговер равномерный, мутация — побитовый сдвиг
и гауссовская для непрерывных генов, элитизм — 1 особь.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from core.problem import Problem
from .base import Encoding, OptimizeResult, Optimizer, ProgressCallback


class GeneticAlgorithm(Optimizer):
    """Генетический алгоритм (популяционная эвристика).

    Параметры: population_size/generations — размер популяции и число
    поколений; crossover_rate/mutation_rate — вероятности операторов;
    tournament_size — размер турнира селекции; elitism — число лучших
    особей, проходящих в следующее поколение без изменений.
    """

    name = "Генетический алгоритм"

    def __init__(
        self,
        population_size: int = 50,
        generations: int = 100,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.05,
        tournament_size: int = 3,
        elitism: int = 1,
    ):
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism = elitism

    def optimize(
        self,
        problem: Problem,
        callback: Optional[ProgressCallback] = None,
        seed: Optional[int] = None,
    ) -> OptimizeResult:
        """Поколенный цикл: селекция → кроссинговер → мутация → оценка;
        callback/seed — см. Optimizer."""
        rng = np.random.default_rng(seed)
        enc = Encoding(problem)
        n = enc.n
        pop = np.array([enc.random_vector(rng) for _ in range(self.population_size)])
        fitness = np.array([problem.evaluate(enc.decode(v)).objective for v in pop])

        def make_result(best_idx: int, history: list[float], it: int, stopped: bool) -> OptimizeResult:
            return OptimizeResult(
                algorithm=self.name,
                best=enc.decode(pop[best_idx]),
                eval_result=problem.evaluate(enc.decode(pop[best_idx])),
                history=history,
                iterations=it,
                stopped=stopped,
            )

        history: list[float] = []
        best_idx = int(np.argmax(fitness))
        mode = problem.config.mode
        for gen in range(1, self.generations + 1):
            # элитизм: лучшие особи переносятся в новое поколение как есть
            order = np.argsort(fitness)[::-1]
            new_pop = [pop[i] for i in order[: self.elitism]]
            while len(new_pop) < self.population_size:
                p1 = self._tournament(pop, fitness, rng)
                p2 = self._tournament(pop, fitness, rng)
                c1, c2 = self._crossover(p1, p2, rng) if n else (p1, p2)
                new_pop.append(self._mutate(c1, rng, mode))
                if len(new_pop) < self.population_size:
                    new_pop.append(self._mutate(c2, rng, mode))
            pop = np.array(new_pop)
            fitness = np.array([problem.evaluate(enc.decode(v)).objective for v in pop])
            best_idx = int(np.argmax(fitness))
            history.append(float(fitness[best_idx]))
            if callback is not None and not callback(gen, float(fitness[best_idx])):
                return make_result(best_idx, history, gen, True)
        return make_result(best_idx, history, self.generations, False)

    def _tournament(self, pop: np.ndarray, fitness: np.ndarray, rng) -> np.ndarray:
        """Турнирная селекция: из tournament_size случайных особей — победитель
        по пригодности."""
        idx = rng.integers(0, len(pop), size=self.tournament_size)
        return pop[idx[int(np.argmax(fitness[idx]))]]

    def _crossover(self, p1: np.ndarray, p2: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray]:
        """Равномерный кроссинговер: с вероятностью crossover_rate потомки
        получают гены от обоих родителей по случайной маске, иначе — копии."""
        if rng.random() >= self.crossover_rate:
            return p1.copy(), p2.copy()
        mask = rng.random(len(p1)) < 0.5
        c1 = np.where(mask, p1, p2)
        c2 = np.where(mask, p2, p1)
        return c1, c2

    def _mutate(self, child: np.ndarray, rng, mode: str) -> np.ndarray:
        """Мутация гена с вероятностью mutation_rate: переключение 0↔1
        (дискретный), гауссовское возмущение объёма (непрерывный) или одно из
        двух (смешанный режим)."""
        child = child.copy()
        for i in range(len(child)):
            if rng.random() < self.mutation_rate:
                if mode == "discrete":
                    child[i] = 1.0 - child[i]
                elif mode == "continuous":
                    child[i] = float(np.clip(child[i] + rng.normal(0, 0.2), 0, 1))
                else:
                    if child[i] > 0.05 and rng.random() < 0.5:
                        child[i] = 0.0  # отключение
                    else:
                        child[i] = float(np.clip(child[i] + rng.normal(0, 0.2), 0, 1))
        return child
