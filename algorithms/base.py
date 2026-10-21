"""Общий интерфейс оптимизаторов и кодирование решения."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from core.model import Solution
from core.problem import EvalResult, Problem

# callback(iteration, best_objective) -> bool; вернуть False — остановить
ProgressCallback = Callable[[int, float], bool]


@dataclass
class OptimizeResult:
    """Результат работы алгоритма.

    Атрибуты: algorithm — имя алгоритма; best — лучшее найденное решение x;
    eval_result — оценка решения (EvalResult); history — лучшее значение
    целевой функции по итерациям; iterations — число итераций;
    evaluations — число оценок целевой функции; runtime — время, с;
    stopped — остановлен ли по сигналу callback'а.
    """

    algorithm: str
    best: Solution
    eval_result: EvalResult
    history: list[float] = field(default_factory=list)  # лучшее значение по итерациям
    iterations: int = 0
    evaluations: int = 0
    runtime: float = 0.0
    stopped: bool = False


class Encoding:
    """Кодирование решения вещественным вектором [0, 1]^n.

    Дискретная часть — первые n дискретных кандидатов (порог 0.5),
    непрерывная — объёмы резервирования.
    """

    def __init__(self, problem: Problem):
        self.problem = problem
        self.n = problem.n

    def decode(self, vec: np.ndarray) -> Solution:
        """Вектор [0, 1]^n → решение x согласно режиму переменных:
        discrete — ген > 0.5 означает выбранного кандидата (объём = 1);
        continuous — гены суть объёмы резервирования; mixed — ген > 0.05
        трактуется как выбранный кандидат с объёмом = значению гена."""
        vec = np.clip(np.asarray(vec, dtype=float), 0.0, 1.0)
        mode = self.problem.config.mode
        if mode == "discrete":
            discrete = vec > 0.5  # порог дискретизации выбора кандидата
            continuous = np.ones(self.n)
        elif mode == "continuous":
            discrete = np.ones(self.n, dtype=bool)
            continuous = vec
        else:  # mixed: положительный объём = кандидат выбран (порог 0.05)
            discrete = vec > 0.05
            continuous = vec
        return Solution(discrete=discrete, continuous=continuous)

    def encode(self, sol: Solution) -> np.ndarray:
        """Обратное преобразование: решение x → вектор [0, 1]^n
        (дискретные гены — 0/1, непрерывные — объёмы)."""
        if self.problem.config.mode == "discrete":
            return sol.discrete.astype(float)
        if self.problem.config.mode == "continuous":
            return sol.continuous.astype(float)
        return np.where(sol.discrete, sol.continuous, 0.0)

    def random_vector(self, rng: np.random.Generator) -> np.ndarray:
        """Случайная стартовая хромосома: в дискретном режиме каждый ген = 1
        с вероятностью 0.3; в смешанном невыбранным кандидатам — нулевой
        объём (иначе порог 0.05 засчитал бы шум как выбор)."""
        mode = self.problem.config.mode
        if mode == "discrete":
            return (rng.random(self.n) < 0.3).astype(float)
        if mode == "continuous":
            return rng.random(self.n)
        vec = rng.random(self.n)
        mask = rng.random(self.n) >= 0.3
        vec[mask] = 0.0  # выключенным кандидатам — нулевой объём
        return vec


class Optimizer(ABC):
    """Базовый класс оптимизатора: общий интерфейс optimize() для всех
    алгоритмов (п. 5–6 doc/optim.md максимизируют скалярный objective)."""

    name: str = "base"

    @abstractmethod
    def optimize(
        self,
        problem: Problem,
        callback: Optional[ProgressCallback] = None,
        seed: Optional[int] = None,
    ) -> OptimizeResult:
        """Запустить оптимизацию. callback(iteration, best_objective) вызывается
        на каждой итерации; возврат False — мягкая остановка. seed фиксирует
        ГСЧ (None — нет). Возвращает OptimizeResult."""
        ...

    def _run(self, problem: Problem, body: Callable[[], OptimizeResult]) -> OptimizeResult:
        """Обёртка над телом алгоритма: замер времени, имя алгоритма и число
        реальных оценок целевой функции."""
        t0 = time.perf_counter()
        result = body()
        result.runtime = time.perf_counter() - t0
        result.algorithm = self.name
        result.evaluations = problem.evaluations
        return result
