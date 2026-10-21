"""Загрузка/сохранение данных (JSON/CSV), экспорт результатов и
генерация сценариев поражения."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

from .model import Candidate, Demand, Link, Network, Node, Scenario


def load_network(path: str | Path) -> Network:
    """Загрузить сеть из JSON (ключи nodes/links/demands/candidates)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Network.from_dict(data)


def save_network(net: Network, path: str | Path) -> None:
    """Сохранить сеть в JSON (см. load_network)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(net.to_dict(), f, ensure_ascii=False, indent=2)


def load_scenarios(path: str | Path) -> list[Scenario]:
    """Загрузить сценарии из JSON — объект {"scenarios": [...]} или простой
    список сценариев."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Scenario.from_dict(x) for x in data.get("scenarios", data)]


def save_scenarios(scenarios: list[Scenario], path: str | Path) -> None:
    """Сохранить сценарии в JSON вида {"scenarios": [...]}."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {"scenarios": [s.to_dict() for s in scenarios]},
            f,
            ensure_ascii=False,
            indent=2,
        )


def load_example(path: str | Path) -> tuple[Network, list[Scenario]]:
    """Загрузить файл примера: сеть + сценарии в одном JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    net = Network.from_dict(data)
    scenarios = [Scenario.from_dict(x) for x in data.get("scenarios", [])]
    return net, scenarios


def save_example(net: Network, scenarios: list[Scenario], path: str | Path) -> None:
    """Сохранить пример: сеть + сценарии в одном JSON (см. load_example)."""
    data = net.to_dict()
    data["scenarios"] = [s.to_dict() for s in scenarios]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# CSV: сеть хранится набором таблиц (nodes/links/demands/candidates) в одном
# каталоге, сценарии — одним CSV-файлом.
# ---------------------------------------------------------------------------


def _write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    """Записать CSV с заголовком header и строками rows (UTF-8)."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Прочитать CSV целиком как список словарей по заголовку."""
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def save_network_csv(net: Network, directory: str | Path) -> None:
    """Экспортировать сеть в каталог directory четырьмя файлами:
    nodes.csv, links.csv, demands.csv, candidates.csv."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    _write_csv(d / "nodes.csv", ["id", "kind"], [[n.id, n.kind] for n in net.nodes.values()])
    _write_csv(
        d / "links.csv",
        ["source", "target", "capacity", "cost", "delay", "reliability"],
        [[l.source, l.target, l.capacity, l.cost, l.delay, l.reliability] for l in net.links],
    )
    _write_csv(
        d / "demands.csv",
        ["source", "target", "volume"],
        [[dm.source, dm.target, dm.volume] for dm in net.demands],
    )
    _write_csv(
        d / "candidates.csv",
        ["id", "source", "target", "capacity", "cost"],
        [[c.id, c.source, c.target, c.capacity, c.cost] for c in net.candidates],
    )


def load_network_csv(directory: str | Path) -> Network:
    """Загрузить сеть из CSV-каталога (см. save_network_csv); отсутствующие
    таблицы трактуются как пустые."""
    d = Path(directory)

    def read(name: str) -> list[dict[str, str]]:
        p = d / name
        return _read_csv(p) if p.exists() else []

    net = Network(
        nodes={r["id"]: Node(id=r["id"], kind=r.get("kind") or "router") for r in read("nodes.csv")},
        links=[Link.from_dict(r) for r in read("links.csv")],
        demands=[Demand.from_dict(r) for r in read("demands.csv")],
        candidates=[Candidate.from_dict(r) for r in read("candidates.csv")],
    )
    net.validate()
    return net


def save_scenarios_csv(scenarios: list[Scenario], path: str | Path) -> None:
    """Сохранить сценарии в CSV: множества отказов кодируются строками
    через «;», каналы — в формате ИСТОЧНИК-НАЗНАЧЕНИЕ."""
    _write_csv(
        Path(path),
        ["id", "p", "failed_nodes", "failed_links"],
        [
            [
                s.id,
                s.p,
                ";".join(sorted(s.failed_nodes)),
                ";".join(f"{a}-{b}" for a, b in sorted(s.failed_links)),
            ]
            for s in scenarios
        ],
    )


def load_scenarios_csv(path: str | Path) -> list[Scenario]:
    """Загрузить сценарии из CSV (см. save_scenarios_csv)."""
    out: list[Scenario] = []
    for r in _read_csv(Path(path)):
        nodes = {x for x in r.get("failed_nodes", "").split(";") if x}
        links: set[tuple[str, str]] = set()
        for item in r.get("failed_links", "").split(";"):
            if item and "-" in item:
                a, _, b = item.partition("-")
                links.add((a, b))
        out.append(Scenario(id=r["id"], p=float(r["p"]), failed_nodes=nodes, failed_links=links))
    return out


def export_result_csv(rows: list[dict[str, object]], path: str | Path) -> None:
    """Экспорт сравнительной таблицы результатов (ключи первого словаря —
    заголовки столбцов)."""
    if not rows:
        return
    header = list(rows[0].keys())
    _write_csv(Path(path), header, [[r.get(h, "") for h in header] for r in rows])


def save_json(data: dict, path: str | Path) -> None:
    """Сохранить произвольный словарь в JSON (UTF-8, отступы 2)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def generate_scenarios(
    net: Network,
    count: int = 8,
    p_link: float = 0.15,
    p_node: float = 0.05,
    seed: int | None = None,
) -> list[Scenario]:
    """Автогенерация случайных сценариев отказа: каждый канал отказывает
    с вероятностью p_link, каждый узел — с вероятностью p_node. Штатный режим
    (без отказов) не генерируется."""
    rng = random.Random(seed)
    scenarios: list[Scenario] = []
    for i in range(count):
        failed_nodes = {n for n in net.nodes if rng.random() < p_node}
        failed_links = {l.key for l in net.links if rng.random() < p_link}
        scenarios.append(
            Scenario(id=f"S{i+1}", p=1.0 / count, failed_nodes=failed_nodes, failed_links=failed_links)
        )
    return scenarios
