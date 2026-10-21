"""Метод имитации отжига (глава 3 doc/optim.md).

Соседнее решение: случайное изменение одного гена — переключение дискретного
кандидата или гауссовское возмущение объёма. Критерий Метрополиса,
геометрическое охлаждение T_k = T_0 · cooling^k.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from core.problem import Problem
from .base import Encoding, OptimizeResult, Optimizer, ProgressCallback


class SimulatedAnnealing(Optimizer):
    """Имитация отжига (стохастический локальный поиск).

    Параметры: t0 — начальная температура T₀; cooling — множитель
    геометрического охлаждения (0 < cooling < 1); iterations — предел
    итераций; t_min — остановка при температуре ниже порога.
    """

    name = "Имитация отжига"

    def __init__(
        self,
        t0: float = 1.0,
        cooling: float = 0.98,
        iterations: int = 2000,
        t_min: float = 1e-3,
    ):
        self.t0 = t0
        self.cooling = cooling
        self.iterations = iterations
        self.t_min = t_min

    def optimize(
        self,
        problem: Problem,
        callback: Optional[ProgressCallback] = None,
        seed: Optional[int] = None,
    ) -> OptimizeResult:
        """Прогон отжига: цикл «сосед → критерий Метрополиса → охлаждение»
        до исчерпания итераций или температуры; callback/seed — см. Optimizer."""
        rng = np.random.default_rng(seed)
        enc = Encoding(problem)
        cur = enc.random_vector(rng)
        cur_fit = problem.evaluate(enc.decode(cur)).objective
        best, best_fit = cur.copy(), cur_fit
        history = [best_fit]

        def make_result(it: int, stopped: bool) -> OptimizeResult:
            return OptimizeResult(
                algorithm=self.name,
                best=enc.decode(best),
                eval_result=problem.evaluate(enc.decode(best)),
                history=history,
                iterations=it,
                stopped=stopped,
            )

        mode = problem.config.mode
        t = self.t0
        it = 0
        while it < self.iterations and t > self.t_min:
            it += 1
            nxt = self._neighbor(cur, rng, mode)
            fit = problem.evaluate(enc.decode(nxt)).objective
            delta = fit - cur_fit
            # критерий Метрополиса: улучшение (ΔF ≥ 0) принимаем всегда,
            # ухудшение — с вероятностью exp(ΔF/T)
            if delta >= 0 or rng.random() < np.exp(delta / max(t, 1e-12)):
                cur, cur_fit = nxt, fit
                if fit > best_fit:
                    best, best_fit = nxt.copy(), fit
            history.append(best_fit)
            t *= self.cooling  # геометрическое охлаждение T_{k+1} = T_k · cooling
            if callback is not None and not callback(it, float(best_fit)):
                return make_result(it, True)
        return make_result(it, False)

    def _neighbor(self, vec: np.ndarray, rng, mode: str) -> np.ndarray:
        """Соседнее решение: изменение одного случайного гена — переключение
        выбора (дискретный) или гауссовское возмущение объёма (непрерывный/
        смешанный, с вероятностью обнуления)."""
        nxt = vec.copy()
        i = int(rng.integers(0, len(vec)))
        if mode == "discrete":
            nxt[i] = 1.0 - nxt[i]
        elif mode == "continuous":
            nxt[i] = float(np.clip(nxt[i] + rng.normal(0, 0.2), 0, 1))
        else:
            if nxt[i] > 0.05 and rng.random() < 0.5:
                nxt[i] = 0.0
            else:
                nxt[i] = float(np.clip(nxt[i] + rng.normal(0, 0.2), 0, 1))
        return nxt
