"""Метод ветвей и границ для малых экземпляров (глава 3 doc/optim.md).

Применим к дискретной постановке 5.1 (максимизация Ψ при бюджетном
ограничении): Ψ монотонно не убывает по добавлению резервных каналов,
поэтому верхняя граница узла — значение Ψ на полном завершении
(все свободные кандидаты включены). Ограничения в границу не входят
(релаксация), штраф лишь уменьшает итоговую целевую функцию.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from core.problem import Problem
from .base import Encoding, OptimizeResult, Optimizer, ProgressCallback


class BranchAndBound(Optimizer):
    """Ветви и границы: точный перебор с отсечением ветвей по верхней границе.

    Применим только к постановке 5.1 (максимизация Ψ) в дискретном режиме;
    max_nodes — предел числа узлов дерева перебора (страховка от взрыва).
    """

    name = "Ветви и границы"

    def __init__(self, max_nodes: int = 200_000):
        self.max_nodes = max_nodes

    def optimize(
        self,
        problem: Problem,
        callback: Optional[ProgressCallback] = None,
        seed: Optional[int] = None,
    ) -> OptimizeResult:
        """Полный обход дерева решений в глубину с отсечением; возвращает
        точно оптимальный вектор x (при неисчерпании max_nodes)."""
        cfg = problem.config
        if cfg.formulation != "max_survivability":
            raise ValueError(
                "Ветви и границы реализованы для постановки 5.1 "
                "(максимизация живучести при бюджетном ограничении)"
            )
        if cfg.mode != "discrete":
            raise ValueError("Ветви и границы работают в дискретном режиме переменных")

        enc = Encoding(problem)
        n = enc.n
        rng = np.random.default_rng(seed)
        del rng  # детерминированный алгоритм; параметр оставлен для интерфейса

        costs = np.array([c.cost for c in problem.network.candidates])
        budget = cfg.budget

        state = {
            "best_psi": -np.inf,
            "best_vec": np.zeros(n),
            "nodes": 0,
            "stopped": False,
            "history": [],
            "it": 0,
        }

        def dfs(pos: int, vec: np.ndarray, spent: float) -> None:
            """Обход в глубину: pos — номер разветвляемого кандидата,
            vec — зафиксированные 0/1, spent — уже потраченный бюджет."""
            if state["stopped"]:
                return
            state["nodes"] += 1
            state["it"] += 1
            if state["nodes"] > self.max_nodes:
                state["stopped"] = True
                return

            # верхняя граница: полное завершение всех свободных ветвей
            if pos < n:
                completion = vec.copy()
                completion[pos:] = 1.0
            else:
                completion = vec
            ub = problem.evaluate(enc.decode(completion)).psi
            if ub <= state["best_psi"] + 1e-12:
                return  # отсечение по границе: Ψ не превзойдёт текущий рекорд

            if pos == n:
                # лист дерева: полный вектор — обновляем рекорд
                res = problem.evaluate(enc.decode(vec))
                if res.psi > state["best_psi"]:
                    state["best_psi"] = res.psi
                    state["best_vec"] = vec.copy()
                    state["history"].append(res.psi)
                    if callback is not None and not callback(state["it"], res.psi):
                        state["stopped"] = True
                return

            # ветвь 1: включаем кандидата pos (если бюджет позволяет)
            if spent + costs[pos] <= budget + 1e-9:
                vec1 = vec.copy()
                vec1[pos] = 1.0
                dfs(pos + 1, vec1, spent + costs[pos])
            # ветвь 0: не включаем
            vec0 = vec.copy()
            vec0[pos] = 0.0
            dfs(pos + 1, vec0, spent)

        dfs(0, np.zeros(n), 0.0)

        if not np.isfinite(state["best_psi"]):
            # допустимого решения не нашлось — возвращаем пустое
            state["best_vec"] = np.zeros(n)

        best = enc.decode(state["best_vec"])
        return OptimizeResult(
            algorithm=self.name,
            best=best,
            eval_result=problem.evaluate(best),
            history=state["history"],
            iterations=state["it"],
            stopped=state["stopped"],
        )
