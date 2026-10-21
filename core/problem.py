"""Постановки задачи оптимизации (п. 5–6 doc/optim.md) и скалярная целевая
функция с штрафами за нарушение ограничений."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np

from . import metrics
from .model import CriticalPair, Network, Scenario, Solution

Formulation = Literal["max_survivability", "min_cost", "minimax", "scalarized"]


@dataclass
class ProblemConfig:
    """Параметры постановки задачи."""

    formulation: Formulation = "max_survivability"  # 5.1 / 5.2 / 5.3 / свёртка
    q_min: float = 0.8                 # минимально допустимое качество Q_min
    budget: float = 50.0               # C_бюдж (п. 4.3)
    psi_required: float = 0.9          # Ψ_треб для постановки 5.2
    quality_mode: str = "flow"         # вариант Q(x, s): flow | connectivity
    mode: str = "discrete"             # тип переменных: discrete|continuous|mixed
    # веса аддитивной свёртки α_k (п. 6): живучесть, −стоимость, −время, качество
    weights: tuple[float, float, float, float] = (0.5, 0.2, 0.1, 0.2)
    unit_time: float = 1.0             # параметр модели времени восстановления
    penalty: float = 10.0              # коэффициент штрафа за нарушения


@dataclass
class EvalResult:
    """Результат оценки решения x.

    objective — скалярное значение целевой функции (максимизируется);
    psi — живучесть Ψ(x); cost — C_Σ(x); t_rec — T_восст(x);
    q_mean/q_min_actual — среднее и минимальное качество Q по сценариям;
    feasible — нет ли нарушений ограничений; violations — сами нарушения
    {имя ограничения: величина превышения}."""

    objective: float
    psi: float
    cost: float
    t_rec: float
    q_mean: float
    q_min_actual: float
    feasible: bool
    violations: dict[str, float] = field(default_factory=dict)


class Problem:
    """Задача оптимизации: сеть + сценарии + конфигурация.

    Единая точка входа для алгоритмов — evaluate(x) -> EvalResult;
    скалярное значение objective согласовано с выбранной постановкой:
      5.1 max_survivability — максимизировать Ψ при бюджете и Q ≥ Q_min;
      5.2 min_cost          — минимизировать C_Σ при Ψ ≥ Ψ_треб и λ_ab ≥ r_ab;
      5.3 minimax           — максимизировать min по критическим сценариям Q;
      scalarized            — максимизировать Σ α_k·F̃_k (п. 6).
    Штрафы (большое M) обеспечивают возврат к допустимой области.
    Оценки кэшируются по ключу решения — в ГА/отжиге много повторов.
    """

    def __init__(
        self,
        network: Network,
        scenarios: list[Scenario],
        config: ProblemConfig | None = None,
        critical_pairs: list[CriticalPair] | None = None,
    ):
        self.network = network
        self.scenarios = scenarios
        self.config = config or ProblemConfig()
        self.critical_pairs = critical_pairs or []
        self._cache: dict[tuple, EvalResult] = {}
        self.evaluations = 0  # число реальных (не из кэша) оценок

    # ------------------------------------------------------------------ #

    @property
    def n(self) -> int:
        """Размерность задачи: число кандидатов на резервирование."""
        return len(self.network.candidates)

    def random_solution(self, rng: np.random.Generator) -> Solution:
        """Случайное допустимое по типам решение (стартовая точка эвристик)."""
        n = self.n
        mode = self.config.mode
        if mode == "discrete":
            discrete = rng.random(n) < 0.3
            continuous = np.ones(n)
        elif mode == "continuous":
            discrete = np.ones(n, dtype=bool)
            continuous = rng.random(n)
        else:  # mixed
            discrete = rng.random(n) < 0.3
            continuous = rng.random(n)
        return Solution(discrete, continuous)

    def _key(self, x: Solution) -> tuple:
        # округление непрерывной части до 3 знаков — ключ кэша оценок
        cont = np.round(x.continuous, 3) if x.continuous.size else ()
        return (tuple(x.discrete.tolist()), tuple(cont.tolist()))

    # ------------------------------------------------------------------ #

    def evaluate(self, x: Solution) -> EvalResult:
        """Оценить решение x (с кэшем по ключу): вернуть EvalResult."""
        key = self._key(x)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        self.evaluations += 1
        res = self._evaluate_uncached(x)
        self._cache[key] = res
        return res

    def _evaluate_uncached(self, x: Solution) -> EvalResult:
        """Собственно вычисление критериев и целевой функции (п. 5–6):
        сначала Q(x, s) по всем сценариям и агрегаты Ψ, C_Σ, T_восст,
        затем сбор нарушений ограничений, затем objective постановки
        минус квадратичный штраф за нарушения."""
        cfg = self.config
        net, scen = self.network, self.scenarios
        vols = x.volumes()
        # в чисто дискретном случае объём фиксирован = 1
        if cfg.mode == "discrete":
            vols = x.discrete.astype(float)
            x = Solution(x.discrete, vols)

        # критерии: качество по сценариям, живучесть, стоимость, время восст.
        qs = [metrics.quality(net, x, s, cfg.quality_mode) for s in scen]
        psi = float(sum(s.p * metrics.phi(q, cfg.q_min) for s, q in zip(scen, qs)))
        c = metrics.cost(net, x)
        t = metrics.recovery_time(net, x, scen, cfg.unit_time)
        q_mean = float(np.mean(qs)) if qs else 1.0
        q_min_actual = float(min(qs)) if qs else 1.0

        # нарушения ограничений (п. 4–5): бюджет, порог Q_min, резервирование
        # приоритетных пар λ_ab ≥ r_ab и (для 5.2) Ψ ≥ Ψ_треб
        violations: dict[str, float] = {}
        if c > cfg.budget + 1e-9:
            violations["budget"] = c - cfg.budget
        q_bad = sum(max(0.0, cfg.q_min - q) for q in qs)
        if q_bad > 0:
            violations["quality"] = q_bad
        conn_bad = 0
        for pair in self.critical_pairs:
            for s in scen:
                lam = metrics.connectivity(net, x, s, pair)
                if lam < pair.r:
                    conn_bad += pair.r - lam
        if conn_bad:
            violations["connectivity"] = float(conn_bad)
        if cfg.formulation == "min_cost" and psi + 1e-9 < cfg.psi_required:
            violations["psi_required"] = cfg.psi_required - psi

        feasible = not violations

        # «сырая» целевая функция постановки (максимизируется)
        if cfg.formulation == "max_survivability":
            raw = psi
        elif cfg.formulation == "min_cost":
            raw = -c  # минимизация стоимости = максимизация −C_Σ
        elif cfg.formulation == "minimax":
            raw = q_min_actual  # гарантированное качество в худшем сценарии
        else:  # scalarized — нормированные критерии «больше = лучше»
            w = cfg.weights  # аддитивная свёртка Φ = Σ α_k·F̃_k (п. 6)
            n_psi = psi
            n_cost = metrics.minmax_normalize(-c, -cfg.budget, 0.0)
            t_max = max(cfg.unit_time, 1e-9)
            n_time = metrics.minmax_normalize(-t, -t_max, 0.0)
            n_q = metrics.minmax_normalize(q_mean, 0.0, 1.0)
            raw = w[0] * n_psi + w[1] * n_cost + w[2] * n_time + w[3] * n_q

        # штраф M·Σ (нарушение)² возвращает поиск в допустимую область
        penalty = cfg.penalty * sum(v * v for v in violations.values())
        return EvalResult(
            objective=raw - penalty,
            psi=psi,
            cost=c,
            t_rec=t,
            q_mean=q_mean,
            q_min_actual=q_min_actual,
            feasible=feasible,
            violations=violations,
        )

    # ------------------------------------------------------------------ #

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Границы вектора кодирования [0, 1]^n."""
        return np.zeros(self.n), np.ones(self.n)
