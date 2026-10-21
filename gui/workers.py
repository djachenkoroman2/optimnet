"""Фоновые потоки оптимизации: интерфейс не блокируется, в GUI летят
сигналы прогресса (итерация, лучшее значение) и результат."""

from __future__ import annotations

import time

from PyQt6.QtCore import QThread, pyqtSignal

from algorithms.base import OptimizeResult
from algorithms.registry import OptimizerSpec, create_optimizer
from core.logging_setup import get_logger
from core.problem import Problem


def _run_timed(spec: OptimizerSpec, problem: Problem, params: dict,
               callback, seed) -> OptimizeResult:
    """Запуск с заполнением runtime и числа оценок (дельта по problem)."""
    evals_before = problem.evaluations
    t0 = time.perf_counter()
    opt = create_optimizer(spec, **params)
    res = opt.optimize(problem, callback=callback, seed=seed)
    if res.runtime <= 0.0:
        res.runtime = time.perf_counter() - t0
    if res.evaluations <= 0:
        res.evaluations = problem.evaluations - evals_before
    return res


class OptimizeWorker(QThread):
    """Один прогон одного алгоритма в фоновом потоке.

    Сигналы: progressed(итерация, лучшее значение) — ход оптимизации;
    finished_ok(OptimizeResult) — успешное завершение; failed(str) — ошибка.
    Общение с GUI только через сигналы — прямой доступ к виджетам из
    потока запрещён Qt.
    """

    progressed = pyqtSignal(int, float)   # итерация, лучшее значение
    finished_ok = pyqtSignal(object)      # OptimizeResult
    failed = pyqtSignal(str)

    def __init__(self, spec: OptimizerSpec, problem: Problem, params: dict,
                 seed: int | None = None, parent=None):
        super().__init__(parent)
        self.spec = spec
        self.problem = problem
        self.params = params
        self.seed = seed
        self._stop = False
        self.log = get_logger("worker")

    def stop(self) -> None:
        """Запросить мягкую остановку: флаг читается в callback, алгоритм
        завершается после текущей итерации."""
        self._stop = True

    def run(self) -> None:
        """Тело потока: запуск алгоритма; каждая итерация шлёт progressed,
        по завершении — finished_ok (или failed при исключении)."""
        self.log.info("Старт %s (параметры %s, seed=%s)", self.spec.name, self.params, self.seed)
        try:
            def cb(it: int, best: float) -> bool:
                self.progressed.emit(it, best)
                return not self._stop

            res = _run_timed(self.spec, self.problem, self.params, cb, self.seed)
            self.log.info(
                "Финиш %s: objective=%.6f, Ψ=%.6f, итераций=%d, оценок=%d",
                self.spec.name, res.eval_result.objective, res.eval_result.psi,
                res.iterations, res.evaluations,
            )
            self.finished_ok.emit(res)
        except Exception as exc:  # noqa: BLE001 — показываем любую ошибку в GUI
            self.log.exception("Ошибка в %s", self.spec.name)
            self.failed.emit(str(exc))


class ExperimentWorker(QThread):
    """Поочерёдный прогон нескольких алгоритмов на одном наборе данных.

    Сигналы: progressed(spec_key, iteration, best); finished_ok(dict);
    algorithm_done(spec_key, OptimizeResult) — после каждого алгоритма.
    """

    progressed = pyqtSignal(str, int, float)
    algorithm_done = pyqtSignal(str, object)
    finished_ok = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, specs: list[OptimizerSpec], problem: Problem,
                 params_by_key: dict[str, dict], seeds: dict[str, list[int]],
                 parent=None):
        super().__init__(parent)
        self.specs = specs
        self.problem = problem
        self.params_by_key = params_by_key
        self.seeds = seeds
        self._stop = False
        self.log = get_logger("experiment")

    def stop(self) -> None:
        """Запросить мягкую остановку всех оставшихся прогонов."""
        self._stop = True

    def run(self) -> None:
        """Цикл по алгоритмам и seed'ам: каждый прогон через _run_timed,
        лучший результат алгоритма — через algorithm_done, всё вместе — через
        finished_ok (ошибки — через failed, цикл прерывается)."""
        results: dict[str, list] = {}
        try:
            for spec in self.specs:
                if self._stop:
                    break
                runs = self.seeds.get(spec.key) or [None]
                per_algorithm = []
                for seed in runs:
                    if self._stop:
                        break

                    def cb(it: int, best: float, key: str = spec.key) -> bool:
                        self.progressed.emit(key, it, best)
                        return not self._stop

                    res = _run_timed(spec, self.problem,
                                     self.params_by_key.get(spec.key, {}), cb, seed)
                    per_algorithm.append(res)
                results[spec.key] = per_algorithm
                if per_algorithm:
                    best = max(per_algorithm, key=lambda r: r.eval_result.objective)
                    self.algorithm_done.emit(spec.key, best)
                self.log.info("Алгоритм %s завершён: прогонов=%d", spec.name, len(per_algorithm))
            self.finished_ok.emit(results)
        except Exception as exc:  # noqa: BLE001
            self.log.exception("Ошибка эксперимента")
            self.failed.emit(str(exc))
