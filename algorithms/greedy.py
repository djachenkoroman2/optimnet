"""Жадная эвристика: последовательное добавление кандидатов с наибольшим
приростом целевой функции на единицу стоимости, пока есть бюджет."""

from __future__ import annotations

from typing import Optional

import numpy as np

from core.problem import Problem
from .base import Encoding, OptimizeResult, Optimizer, ProgressCallback


class Greedy(Optimizer):
    """Жадная эвристика: на каждом шаге добавляется кандидат с наибольшим
    приростом целевой функции на единицу стоимости, пока есть бюджет и
    выгодные кандидаты. Параметров нет."""

    name = "Жадная эвристика"

    def __init__(self):
        pass

    def optimize(
        self,
        problem: Problem,
        callback: Optional[ProgressCallback] = None,
        seed: Optional[int] = None,
    ) -> OptimizeResult:
        """Последовательное жадное построение решения; callback/seed —
        см. Optimizer (алгоритм детерминирован, seed не используется)."""
        cfg = problem.config
        enc = Encoding(problem)
        n = enc.n
        costs = np.array([c.cost for c in problem.network.candidates])
        budget = cfg.budget

        vec = np.zeros(n)
        best_fit = problem.evaluate(enc.decode(vec)).objective
        history = [best_fit]
        stopped = False
        it = 0

        while True:
            it += 1
            spent = float(costs @ (vec > 0.5))
            # перебираем не взятых кандидатов, укладывающихся в остаток бюджета
            best_rate, best_k = 0.0, -1
            for k in range(n):
                if vec[k] > 0.5 or spent + costs[k] > budget + 1e-9:
                    continue
                trial = vec.copy()
                trial[k] = 1.0
                fit = problem.evaluate(enc.decode(trial)).objective
                gain = fit - best_fit
                if gain > 1e-12:
                    rate = gain / max(costs[k], 1e-12)
                    if rate > best_rate:
                        best_rate, best_k = rate, k
            if best_k < 0:
                break  # нет кандидата с положительным приростом — стоп
            vec[best_k] = 1.0
            best_fit = problem.evaluate(enc.decode(vec)).objective
            history.append(best_fit)
            if callback is not None and not callback(it, best_fit):
                stopped = True
                break

        best = enc.decode(vec)
        return OptimizeResult(
            algorithm=self.name,
            best=best,
            eval_result=problem.evaluate(best),
            history=history,
            iterations=it,
            stopped=stopped,
        )
