"""Прогоны алгоритмов для руководства пользователя (реальные числа).

Без GUI: 10 документированных сценариев на примерах small/medium/large и
ручной мини-топологии. Результаты — в doc/results_doc_examples.json
(ensure_ascii=False, indent=2), компактная сводка — в stdout.

Запуск:
    .venv/Scripts/python.exe scripts/run_doc_examples.py

Бюджет на готовых топологиях — как в GUI: 0.5 · Σ стоимостей кандидатов.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import io  # noqa: E402
from core.model import Candidate, Demand, Link, Network, Node, Scenario  # noqa: E402
from core.problem import Problem, ProblemConfig  # noqa: E402

from algorithms.base import OptimizeResult  # noqa: E402
from algorithms.registry import OPTIMIZERS, run_optimizer  # noqa: E402

SPECS = {s.key: s for s in OPTIMIZERS}
OUT_PATH = ROOT / "doc" / "results_doc_examples.json"

# значения по умолчанию из algorithms/registry.py (сигнатуры классов)
GA_DEFAULTS = {
    "population_size": 50, "generations": 100, "crossover_rate": 0.8,
    "mutation_rate": 0.05, "tournament_size": 3, "elitism": 1,
}
SA_DEFAULTS = {"t0": 1.0, "cooling": 0.98, "iterations": 2000, "t_min": 1e-3}
SA_MINIMAX = {"t0": 1.0, "cooling": 0.98, "iterations": 2000, "t_min": 1e-3}
BNB_DEFAULTS = {"max_nodes": 200000}


# --------------------------------------------------------------------- #
# хелперы
# --------------------------------------------------------------------- #

def budget_of(net: Network) -> float:
    """Как в MainWindow._suggest_budget: 0.5 · Σcost, округление до 0.1."""
    return round(0.5 * sum(c.cost for c in net.candidates), 1)


def config_record(cfg: ProblemConfig) -> dict:
    return {
        "formulation": cfg.formulation,
        "mode": cfg.mode,
        "quality_mode": cfg.quality_mode,
        "q_min": cfg.q_min,
        "budget": cfg.budget,
        "psi_required": cfg.psi_required,
        "weights": list(cfg.weights),
        "unit_time": cfg.unit_time,
    }


def run_timed(spec_key: str, problem: Problem, params: dict,
              seed: int | None) -> tuple[OptimizeResult, float, int]:
    """run_optimizer + замер времени и числа новых оценок (дельта по cache)."""
    evals_before = problem.evaluations
    t0 = time.perf_counter()
    res = run_optimizer(SPECS[spec_key], problem, params, seed=seed)
    runtime = time.perf_counter() - t0
    evals = problem.evaluations - evals_before
    return res, runtime, evals


def result_record(res: OptimizeResult, runtime: float, evals: int,
                  net: Network) -> dict:
    er = res.eval_result
    vols = res.best.volumes()
    chosen = [
        {
            "id": c.id,
            "source": c.source,
            "target": c.target,
            "volume": round(float(v), 4),
            "capacity_added": round(float(c.capacity * v), 4),
            "cost": round(float(c.cost * v), 4),
        }
        for c, v in zip(net.candidates, vols) if v > 0
    ]
    return {
        "algorithm": res.algorithm,
        "objective": round(er.objective, 6),
        "psi": round(er.psi, 6),
        "cost": round(er.cost, 6),
        "t_rec": round(er.t_rec, 6),
        "q_mean": round(er.q_mean, 6),
        "q_min_actual": round(er.q_min_actual, 6),
        "feasible": er.feasible,
        "violations": {k: round(v, 6) for k, v in er.violations.items()},
        "iterations": res.iterations,
        "evaluations": evals,
        "runtime_sec": round(runtime, 4),
        "chosen": chosen,
    }


def guard(fn, *args, **kwargs):
    """Выполнить шаг; ошибку зафиксировать в JSON, не прерывая остальные."""
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 — все прогоны должны дойти до конца
        return {"error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()}


# --------------------------------------------------------------------- #
# построитель малой ручной топологии (как tests/conftest.py)
# --------------------------------------------------------------------- #

def build_manual_net() -> tuple[Network, list[Scenario]]:
    net = Network(
        nodes={"A": Node("A", "server"), "B": Node("B", "router"),
               "C": Node("C", "server")},
        links=[
            Link("A", "B", capacity=5.0, cost=2.0),
            Link("B", "C", capacity=5.0, cost=2.0),
            Link("A", "C", capacity=2.0, cost=1.0),
        ],
        demands=[Demand("A", "C", volume=8.0)],
        candidates=[
            Candidate("C1", "A", "B", capacity=10.0, cost=3.0),
            Candidate("C2", "B", "C", capacity=10.0, cost=4.0),
        ],
    )
    scenarios = [
        Scenario("S0", p=0.6),
        Scenario("S1", p=0.4, failed_nodes={"B"}),
    ]
    return net, scenarios


# --------------------------------------------------------------------- #
# 10 сценариев
# --------------------------------------------------------------------- #

def example_1() -> dict:
    """medium, 5.1, flow, discrete, ГА (параметры по умолчанию), seed=42."""
    net, scen = io.load_example(ROOT / "examples" / "medium.json")
    cfg = ProblemConfig(formulation="max_survivability", mode="discrete",
                        quality_mode="flow", q_min=0.8, budget=budget_of(net))
    prob = Problem(net, scen, cfg)
    res, runtime, evals = run_timed("ga", prob, dict(GA_DEFAULTS), seed=42)
    return {
        "id": 1,
        "description": "medium: постановка 5.1 (max_survivability), flow, "
                       "дискретные переменные; ГА, параметры по умолчанию, seed=42",
        "network": "examples/medium.json",
        "config": config_record(cfg),
        "algorithm": {"key": "ga", "name": SPECS["ga"].name, "params": GA_DEFAULTS},
        "seed": 42,
        "result": result_record(res, runtime, evals, net),
    }


def example_2() -> dict:
    """medium, 5.1, flow, discrete, отжиг (t0=1.0, cooling=0.98, it=2000)."""
    net, scen = io.load_example(ROOT / "examples" / "medium.json")
    cfg = ProblemConfig(formulation="max_survivability", mode="discrete",
                        quality_mode="flow", q_min=0.8, budget=budget_of(net))
    prob = Problem(net, scen, cfg)
    res, runtime, evals = run_timed("sa", prob, dict(SA_DEFAULTS), seed=42)
    return {
        "id": 2,
        "description": "medium: постановка 5.1, flow, дискретные переменные; "
                       "имитация отжига (T₀=1.0, cooling=0.98, 2000 итераций, "
                       "T_min=1e-3), seed=42",
        "network": "examples/medium.json",
        "config": config_record(cfg),
        "algorithm": {"key": "sa", "name": SPECS["sa"].name, "params": SA_DEFAULTS},
        "seed": 42,
        "result": result_record(res, runtime, evals, net),
    }


def example_3() -> dict:
    """small, 5.1, flow, discrete, ветви и границы (max_nodes=200000)."""
    net, scen = io.load_example(ROOT / "examples" / "small.json")
    cfg = ProblemConfig(formulation="max_survivability", mode="discrete",
                        quality_mode="flow", q_min=0.8, budget=budget_of(net))
    prob = Problem(net, scen, cfg)
    res, runtime, evals = run_timed("bnb", prob, dict(BNB_DEFAULTS), seed=None)
    return {
        "id": 3,
        "description": "small: постановка 5.1, flow, дискретные переменные; "
                       "ветви и границы (max_nodes=200000) — точное решение",
        "network": "examples/small.json",
        "config": config_record(cfg),
        "algorithm": {"key": "bnb", "name": SPECS["bnb"].name, "params": BNB_DEFAULTS},
        "seed": None,
        "result": result_record(res, runtime, evals, net),
    }


def example_4() -> dict:
    """medium, 5.1, flow, discrete, жадная эвристика (без параметров)."""
    net, scen = io.load_example(ROOT / "examples" / "medium.json")
    cfg = ProblemConfig(formulation="max_survivability", mode="discrete",
                        quality_mode="flow", q_min=0.8, budget=budget_of(net))
    prob = Problem(net, scen, cfg)
    res, runtime, evals = run_timed("greedy", prob, {}, seed=None)
    return {
        "id": 4,
        "description": "medium: постановка 5.1, flow, дискретные переменные; "
                       "жадная эвристика (параметров нет)",
        "network": "examples/medium.json",
        "config": config_record(cfg),
        "algorithm": {"key": "greedy", "name": SPECS["greedy"].name, "params": {}},
        "seed": None,
        "result": result_record(res, runtime, evals, net),
    }


def example_5() -> dict:
    """medium, 5.2 (min_cost), psi_required=0.9, ГА, seed=42."""
    net, scen = io.load_example(ROOT / "examples" / "medium.json")
    cfg = ProblemConfig(formulation="min_cost", mode="discrete",
                        quality_mode="flow", q_min=0.8, budget=budget_of(net),
                        psi_required=0.9)
    prob = Problem(net, scen, cfg)
    res, runtime, evals = run_timed("ga", prob, dict(GA_DEFAULTS), seed=42)
    return {
        "id": 5,
        "description": "medium: постановка 5.2 (min_cost), flow, дискретные "
                       "переменные; Ψ_треб=0.9; ГА, seed=42. Выгодное "
                       "направление — минимизация: objective = −C_Σ со штрафами",
        "network": "examples/medium.json",
        "config": config_record(cfg),
        "algorithm": {"key": "ga", "name": SPECS["ga"].name, "params": GA_DEFAULTS},
        "seed": 42,
        "result": result_record(res, runtime, evals, net),
    }


def example_6() -> dict:
    """medium, 5.3 (minimax), flow, discrete, отжиг, seed=42."""
    net, scen = io.load_example(ROOT / "examples" / "medium.json")
    cfg = ProblemConfig(formulation="minimax", mode="discrete",
                        quality_mode="flow", q_min=0.8, budget=budget_of(net))
    prob = Problem(net, scen, cfg)
    res, runtime, evals = run_timed("sa", prob, dict(SA_MINIMAX), seed=42)
    return {
        "id": 6,
        "description": "medium: постановка 5.3 (minimax — максимин min Q по "
                       "сценариям), flow, дискретные переменные; отжиг, seed=42",
        "network": "examples/medium.json",
        "config": config_record(cfg),
        "algorithm": {"key": "sa", "name": SPECS["sa"].name, "params": SA_MINIMAX},
        "seed": 42,
        "result": result_record(res, runtime, evals, net),
    }


def example_7() -> dict:
    """large, scalarized (свёртка), flow, mode=mixed, ГА, seed=42."""
    net, scen = io.load_example(ROOT / "examples" / "large.json")
    cfg = ProblemConfig(formulation="scalarized", mode="mixed",
                        quality_mode="flow", q_min=0.8, budget=budget_of(net),
                        weights=(0.5, 0.2, 0.1, 0.2))
    prob = Problem(net, scen, cfg)
    res, runtime, evals = run_timed("ga", prob, dict(GA_DEFAULTS), seed=42)
    return {
        "id": 7,
        "description": "large: аддитивная свёртка (scalarized), flow, смешанные "
                       "переменные (mixed); веса (0.5, 0.2, 0.1, 0.2); "
                       "ГА, seed=42",
        "network": "examples/large.json",
        "config": config_record(cfg),
        "algorithm": {"key": "ga", "name": SPECS["ga"].name, "params": GA_DEFAULTS},
        "seed": 42,
        "result": result_record(res, runtime, evals, net),
    }


def example_8() -> dict:
    """medium, 5.1 — эксперимент: ga/sa/greedy × seeds 42,43,44."""
    net, scen = io.load_example(ROOT / "examples" / "medium.json")
    cfg = ProblemConfig(formulation="max_survivability", mode="discrete",
                        quality_mode="flow", q_min=0.8, budget=budget_of(net))
    prob = Problem(net, scen, cfg)
    seeds = [42, 43, 44]
    params_by_key = {"ga": dict(GA_DEFAULTS), "sa": dict(SA_DEFAULTS), "greedy": {}}
    algorithms: dict[str, dict] = {}
    for key in ("ga", "sa", "greedy"):
        runs = []
        for sd in seeds:
            res, runtime, evals = run_timed(key, prob, params_by_key[key], seed=sd)
            runs.append((res, runtime, evals))
        objs = [r.eval_result.objective for r, _, _ in runs]
        best_idx = max(range(len(runs)), key=lambda i: objs[i])
        best, _, _ = runs[best_idx]
        algorithms[key] = {
            "name": SPECS[key].name,
            "params": params_by_key[key],
            "seeds": seeds,
            "objectives": [round(o, 6) for o in objs],
            "best_objective": round(objs[best_idx], 6),
            "mean_objective": round(sum(objs) / len(objs), 6),
            "worst_objective": round(min(objs), 6),
            "psi_of_best": round(best.eval_result.psi, 6),
            "cost_of_best": round(best.eval_result.cost, 6),
            "mean_runtime_sec": round(sum(t for _, t, _ in runs) / len(runs), 4),
            "mean_iterations": round(sum(r.iterations for r, _, _ in runs) / len(runs), 1),
            "mean_evaluations": round(sum(e for _, _, e in runs) / len(runs), 1),
            "history_of_best": [round(float(h), 6) for h in best.history],
            "best_run": result_record(best, runs[best_idx][1], runs[best_idx][2], net),
        }
    return {
        "id": 8,
        "description": "medium: постановка 5.1, flow, дискретные переменные; "
                       "эксперимент — ga/sa/greedy, по 3 прогона (seeds 42, 43, 44)",
        "network": "examples/medium.json",
        "config": config_record(cfg),
        "algorithms": algorithms,
    }


def example_9() -> dict:
    """Ручная малая топология (как в tests/conftest.py), 5.1, ГА, budget=3.5."""
    net, scen = build_manual_net()
    cfg = ProblemConfig(formulation="max_survivability", mode="discrete",
                        quality_mode="flow", q_min=0.8, budget=3.5)
    prob = Problem(net, scen, cfg)
    res, runtime, evals = run_timed("ga", prob, dict(GA_DEFAULTS), seed=42)
    return {
        "id": 9,
        "description": "ручная топология (tests/conftest.py): A(server), "
                       "B(router), C(server); каналы A→B (5, ц.2), B→C (5, ц.2), "
                       "A→C (2, ц.1); потребность A→C объём 8; кандидаты "
                       "C1 (A→B, 10, ц.3), C2 (B→C, 10, ц.4); сценарии "
                       "S0 (p=0.6, без отказов), S1 (p=0.4, отказ узла B). "
                       "Постановка 5.1, flow, дискретные переменные, "
                       "Q_min=0.8, бюджет 3.5; ГА, seed=42",
        "network": "manual (построена кодом, см. description)",
        "config": config_record(cfg),
        "algorithm": {"key": "ga", "name": SPECS["ga"].name, "params": GA_DEFAULTS},
        "seed": 42,
        "result": result_record(res, runtime, evals, net),
    }


def _generated_scenarios(net: Network) -> list[Scenario]:
    """Как gui/tab_scenarios.py::_generate: S0 (p=0.4) + 8 случайных,
    перенормированных до суммы 0.6."""
    new = io.generate_scenarios(net, count=8, p_link=0.15, p_node=0.05, seed=42)
    for s in new:
        s.p = round(0.6 / len(new), 6)
    return [Scenario(id="S0", p=0.4)] + new


def example_10() -> dict:
    """medium, 5.1, quality_mode=connectivity: штатные и сгенерированные
    сценарии, ГА, seed=42."""
    net, scen = io.load_example(ROOT / "examples" / "medium.json")

    cfg = ProblemConfig(formulation="max_survivability", mode="discrete",
                        quality_mode="connectivity", q_min=0.8, budget=budget_of(net))
    prob = Problem(net, scen, cfg)
    res, runtime, evals = run_timed("ga", prob, dict(GA_DEFAULTS), seed=42)
    sub_original = {
        "description": "штатные сценарии из examples/medium.json, "
                       "Q(x,s) — связность (connectivity)",
        "config": config_record(cfg),
        "n_scenarios": len(scen),
        "result": result_record(res, runtime, evals, net),
    }

    gen = _generated_scenarios(net)
    cfg2 = ProblemConfig(formulation="max_survivability", mode="discrete",
                         quality_mode="connectivity", q_min=0.8,
                         budget=budget_of(net))
    prob2 = Problem(net, gen, cfg2)
    res2, runtime2, evals2 = run_timed("ga", prob2, dict(GA_DEFAULTS), seed=42)
    sub_generated = {
        "description": "сгенерированные сценарии: generate_scenarios(net, "
                       "count=8, p_link=0.15, p_node=0.05, seed=42) + штатный "
                       "S0 (p=0.4), остальные перенормированы до суммы 0.6; "
                       "Q(x,s) — связность",
        "config": config_record(cfg2),
        "n_scenarios": len(gen),
        "scenarios": [
            {"id": s.id, "p": s.p, "failed_nodes": sorted(s.failed_nodes),
             "failed_links": sorted(list(k) for k in s.failed_links)}
            for s in gen
        ],
        "result": result_record(res2, runtime2, evals2, net),
    }

    return {
        "id": 10,
        "description": "medium: постановка 5.1, дискретные переменные, "
                       "качество Q(x,s) — connectivity; ГА, seed=42 — два "
                       "подпрогона: штатные и сгенерированные сценарии",
        "network": "examples/medium.json",
        "algorithm": {"key": "ga", "name": SPECS["ga"].name, "params": GA_DEFAULTS},
        "seed": 42,
        "config": sub_original["config"],
        "sub_runs": [sub_original, sub_generated],
    }


EXAMPLES = [example_1, example_2, example_3, example_4, example_5,
            example_6, example_7, example_8, example_9, example_10]


# --------------------------------------------------------------------- #
# сводка в stdout
# --------------------------------------------------------------------- #

def _short(s: str, n: int = 46) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def print_summary(doc: dict) -> None:
    print("\n" + "=" * 118)
    hdr = (f"{'№':>2} {'описание':48s} {'F':>10} {'Ψ':>8} {'C_Σ':>7} "
           f"{'Q̄':>7} {'minQ':>6} {'t,с':>6} {'итер':>6}  выбранные кандидаты")
    print(hdr)
    print("-" * 118)
    for ex in doc["examples"]:
        if "error" in ex:
            print(f"{ex['id']:>2} {'ОШИБКА: ' + ex['error']}")
            continue
        if "algorithms" in ex:  # эксперимент (пример 8)
            for key, alg in ex["algorithms"].items():
                chosen = ", ".join(f"{c['id']}:{c['volume']:g}" for c in alg["best_run"]["chosen"]) or "—"
                print(f"{ex['id']:>2} [{key:<6}] {alg['name'][:38]:40s} "
                      f"{alg['best_objective']:>10.4f} {alg['psi_of_best']:>8.4f} "
                      f"{alg['cost_of_best']:>7.2f} {'—':>7} {'—':>6} "
                      f"{alg['mean_runtime_sec']:>6.3f} {alg['mean_iterations']:>6.1f}  {chosen}")
            continue
        subs = ex.get("sub_runs")
        if subs:  # пример 10
            for sub in subs:
                r = sub["result"]
                chosen = ", ".join(f"{c['id']}:{c['volume']:g}" for c in r["chosen"]) or "—"
                print(f"{ex['id']:>2} {_short(sub['description'], 46):48s} "
                      f"{r['objective']:>10.4f} {r['psi']:>8.4f} {r['cost']:>7.2f} "
                      f"{r['q_mean']:>7.4f} {r['q_min_actual']:>6.3f} "
                      f"{r['runtime_sec']:>6.3f} {r['iterations']:>6d}  {chosen}")
            continue
        r = ex["result"]
        chosen = ", ".join(f"{c['id']}:{c['volume']:g}" for c in r["chosen"]) or "—"
        print(f"{ex['id']:>2} {_short(ex['description'], 46):48s} "
              f"{r['objective']:>10.4f} {r['psi']:>8.4f} {r['cost']:>7.2f} "
              f"{r['q_mean']:>7.4f} {r['q_min_actual']:>6.3f} "
              f"{r['runtime_sec']:>6.3f} {r['iterations']:>6d}  {chosen}")
    print("=" * 118)


# --------------------------------------------------------------------- #

def main() -> int:
    # консоль Windows по умолчанию cp1251 — сводка содержит Ψ/Q̄
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    t_all = time.perf_counter()
    examples: list[dict] = []
    for fn in EXAMPLES:
        ex = guard(fn)
        if "id" not in ex:
            ex = {"id": EXAMPLES.index(fn) + 1, "description": fn.__doc__ or "",
                  **ex}
        examples.append(ex)
        status = "OK" if "error" not in ex else f"ОШИБКА: {ex['error']}"
        print(f"пример {ex['id']:>2}: {status}")

    doc = {
        "meta": {
            "title": "Прогоны алгоритмов OptimNet для руководства пользователя",
            "budget_rule": "0.5 · Σ стоимостей кандидатов (как в GUI)",
            "runtime_total_sec": round(time.perf_counter() - t_all, 3),
        },
        "examples": examples,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    print(f"\nJSON сохранён: {OUT_PATH} "
          f"({OUT_PATH.stat().st_size} байт, примеров: {len(examples)})")

    print_summary(doc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
