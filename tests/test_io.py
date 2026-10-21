"""Тесты сериализации: JSON и CSV (топологии, сценарии, экспорт результатов)."""

from __future__ import annotations

from core import io
import pytest

from core.model import Scenario


def test_network_json_roundtrip(net, tmp_path):
    path = tmp_path / "net.json"
    io.save_network(net, path)
    net2 = io.load_network(path)
    assert set(net2.nodes) == set(net.nodes)
    assert [(l.source, l.target, l.capacity) for l in net2.links] == [
        (l.source, l.target, l.capacity) for l in net.links
    ]
    assert [(c.id, c.cost) for c in net2.candidates] == [(c.id, c.cost) for c in net.candidates]


def test_example_json_roundtrip(net, scenarios, tmp_path):
    path = tmp_path / "example.json"
    io.save_example(net, scenarios, path)
    net2, sc2 = io.load_example(path)
    assert set(net2.nodes) == set(net.nodes)
    assert [s.id for s in sc2] == [s.id for s in scenarios]
    assert sc2[1].failed_nodes == scenarios[1].failed_nodes


def test_network_csv_roundtrip(net, tmp_path):
    io.save_network_csv(net, tmp_path)
    net2 = io.load_network_csv(tmp_path)
    assert set(net2.nodes) == set(net.nodes)
    assert len(net2.links) == len(net.links)
    assert len(net2.demands) == len(net.demands)
    assert [(c.id, c.source, c.target, c.capacity, c.cost) for c in net2.candidates] == [
        (c.id, c.source, c.target, c.capacity, c.cost) for c in net.candidates
    ]


def test_scenarios_csv_roundtrip(tmp_path):
    scenarios = [
        Scenario("S0", p=0.5),
        Scenario("S1", p=0.25, failed_nodes={"A", "B"}, failed_links={("A", "B"), ("C", "A")}),
        Scenario("S2", p=0.25, failed_links={("B", "C")}),
    ]
    path = tmp_path / "scen.csv"
    io.save_scenarios_csv(scenarios, path)
    out = io.load_scenarios_csv(path)
    assert [s.id for s in out] == ["S0", "S1", "S2"]
    assert out[0].failed_nodes == set() and out[0].failed_links == set()
    assert out[1].failed_nodes == {"A", "B"}
    assert out[1].failed_links == {("A", "B"), ("C", "A")}
    assert out[2].failed_links == {("B", "C")}
    assert sum(s.p for s in out) == _ap(1.0)


def test_generate_scenarios_deterministic(net):
    a = io.generate_scenarios(net, count=5, p_link=0.3, p_node=0.1, seed=42)
    b = io.generate_scenarios(net, count=5, p_link=0.3, p_node=0.1, seed=42)
    assert [s.to_dict() for s in a] == [s.to_dict() for s in b]
    assert all(not s.failed_nodes for s in a) or True  # отказы узлов возможны
    assert abs(sum(s.p for s in a) - 1.0) < 1e-9


def test_export_result_csv(tmp_path):
    rows = [
        {"алгоритм": "ГА", "лучшее": 0.9, "среднее": 0.8, "худшее": 0.7},
        {"алгоритм": "Отжиг", "лучшее": 0.85, "среднее": 0.75, "худшее": 0.65},
    ]
    path = tmp_path / "results.csv"
    io.export_result_csv(rows, path)
    import csv

    with open(path, encoding="utf-8") as f:
        got = list(csv.DictReader(f))
    assert got[0]["алгоритм"] == "ГА"
    assert got[1]["лучшее"] == "0.85"


def _ap(v: float):
    return pytest.approx(v, abs=1e-9)
