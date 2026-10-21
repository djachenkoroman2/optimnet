"""Общее состояние приложения: сеть, сценарии, конфигурация задачи,
приоритетные пары и результаты прогона. Вкладки читают и меняют его."""

from __future__ import annotations

from dataclasses import dataclass, field

from core.model import CriticalPair, Network, Scenario
from core.problem import ProblemConfig


@dataclass
class AppState:
    """Общее состояние приложения (читают и меняют все вкладки).

    Атрибуты: network — модель сети; scenarios — сценарии поражения;
    config — параметры постановки задачи; critical_pairs — приоритетные пары
    (a, b, r_ab); last_results — результаты последнего прогона
    {ключ алгоритма: OptimizeResult}.
    """

    network: Network = field(default_factory=Network)
    scenarios: list[Scenario] = field(default_factory=list)
    config: ProblemConfig = field(default_factory=ProblemConfig)
    critical_pairs: list[CriticalPair] = field(default_factory=list)
    # результаты последнего прогона по алгоритмам: {ключ алгоритма: OptimizeResult}
    last_results: dict = field(default_factory=dict)

    def has_data(self) -> bool:
        """True, если есть минимум данных для запуска оптимизации:
        узлы, потребности и хотя бы один сценарий."""
        return bool(self.network.nodes and self.network.demands and self.scenarios)
