"""Реестр оптимизаторов.

UI строит форму параметров алгоритма автоматически по спецификациям из этого
модуля: новый алгоритм достаточно унаследовать от Optimizer и добавить
в OPTIMIZERS — правок в интерфейсе не потребуется.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Optional

from core.problem import Problem

from .base import OptimizeResult, Optimizer, ProgressCallback
from .bnb import BranchAndBound
from .ga import GeneticAlgorithm
from .greedy import Greedy
from .sa import SimulatedAnnealing


@dataclass(frozen=True)
class ParamSpec:
    """Описание одного настраиваемого параметра алгоритма."""

    name: str
    label: str          # подпись на русском (UI)
    kind: str           # "int" | "float"
    default: float
    lo: float
    hi: float
    step: float = 1.0


@dataclass(frozen=True)
class OptimizerSpec:
    """Описание алгоритма для UI и фабрики."""

    key: str
    cls: type[Optimizer]
    params: tuple[ParamSpec, ...]
    formulations: Optional[tuple[str, ...]] = None  # None — все постановки
    modes: Optional[tuple[str, ...]] = None         # None — все режимы переменных

    @property
    def name(self) -> str:
        """Отображаемое имя алгоритма (класс-атрибут cls.name)."""
        return self.cls.name

    def supports(self, formulation: str, mode: str) -> bool:
        """True, если алгоритм применим к постановке formulation и режиму
        переменных mode (None в спецификации — без ограничения)."""
        f_ok = self.formulations is None or formulation in self.formulations
        m_ok = self.modes is None or mode in self.modes
        return f_ok and m_ok


_LABELS = {
    "population_size": "Размер популяции",
    "generations": "Число поколений",
    "crossover_rate": "Вероятность кроссинговера",
    "mutation_rate": "Вероятность мутации",
    "tournament_size": "Размер турнира",
    "elitism": "Элитизм (особей)",
    "t0": "Начальная температура T₀",
    "cooling": "Скорость охлаждения",
    "iterations": "Число итераций",
    "t_min": "Минимальная температура",
    "max_nodes": "Макс. число узлов дерева",
}

# разумные границы для автоматически разбираемых параметров
_BOUNDS = {
    "population_size": (4, 500, 1, "int"),
    "generations": (1, 5000, 1, "int"),
    "crossover_rate": (0.0, 1.0, 0.05, "float"),
    "mutation_rate": (0.0, 1.0, 0.01, "float"),
    "tournament_size": (2, 20, 1, "int"),
    "elitism": (0, 10, 1, "int"),
    "t0": (1e-3, 1000.0, 0.1, "float"),
    "cooling": (0.5, 0.999, 0.001, "float"),
    "iterations": (10, 1_000_000, 10, "int"),
    "t_min": (1e-12, 1.0, 0.001, "float"),
    "max_nodes": (100, 10_000_000, 100, "int"),
}


def _params_from_signature(cls: type[Optimizer]) -> tuple[ParamSpec, ...]:
    """Разобрать сигнатуру __init__ класса алгоритма в спецификации параметров:
    только аргументы с значениями по умолчанию; границы/шаг/тип — из _BOUNDS,
    подписи — из _LABELS (нет в словарях — разумные значения и имя аргумента)."""
    sig = inspect.signature(cls.__init__)
    specs: list[ParamSpec] = []
    for name, p in sig.parameters.items():
        if name == "self" or p.default is inspect.Parameter.empty:
            continue
        lo, hi, step, kind = _BOUNDS.get(name, (0.0, 1000.0, 1.0, "int"))
        specs.append(
            ParamSpec(
                name=name,
                label=_LABELS.get(name, name),
                kind=kind,
                default=float(p.default),
                lo=lo,
                hi=hi,
                step=step,
            )
        )
    return tuple(specs)


OPTIMIZERS: tuple[OptimizerSpec, ...] = (
    # ветви и границы — только постановка 5.1 и дискретный режим
    OptimizerSpec("ga", GeneticAlgorithm, _params_from_signature(GeneticAlgorithm)),
    OptimizerSpec("sa", SimulatedAnnealing, _params_from_signature(SimulatedAnnealing)),
    OptimizerSpec("bnb", BranchAndBound, _params_from_signature(BranchAndBound),
                  formulations=("max_survivability",), modes=("discrete",)),
    OptimizerSpec("greedy", Greedy, _params_from_signature(Greedy)),
)


def available_optimizers(formulation: str | None = None, mode: str | None = None) -> list[OptimizerSpec]:
    """Список алгоритмов, применимых к постановке/режиму (без фильтра — все)."""
    out = []
    for spec in OPTIMIZERS:
        if formulation is None and mode is None:
            out.append(spec)
        elif formulation is not None and mode is not None and spec.supports(formulation, mode):
            out.append(spec)
    return out


def create_optimizer(spec: OptimizerSpec, **kwargs) -> Optimizer:
    """Создать экземпляр алгоритма; неизвестные/выключенные параметры игнорируются."""
    allowed = {p.name: p for p in spec.params}
    clean = {k: (int(v) if allowed[k].kind == "int" else float(v)) for k, v in kwargs.items() if k in allowed}
    return spec.cls(**clean)


def run_optimizer(
    spec: OptimizerSpec,
    problem: Problem,
    params: dict | None = None,
    callback: ProgressCallback | None = None,
    seed: int | None = None,
) -> OptimizeResult:
    """Удобная обёртка: создать алгоритм и запустить."""
    return create_optimizer(spec, **(params or {})).optimize(problem, callback=callback, seed=seed)


__all__ = [
    "OPTIMIZERS",
    "OptimizerSpec",
    "ParamSpec",
    "available_optimizers",
    "create_optimizer",
    "run_optimizer",
]
