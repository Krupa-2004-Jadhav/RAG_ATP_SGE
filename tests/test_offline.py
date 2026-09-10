"""Offline tests - zero tokens. Run before any real LLM spend (CLAUDE.md §15)."""

import csv
import json
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from core.llm_client import MockLLMClient
from core.optima import brute_force_knapsack, brute_force_tsp, held_karp, knapsack_dp
from methods import pipeline, registry
from problems.knapsack import KnapsackProblem
from problems.vrp import VRPProblem
from run.run_experiment import load_completed_keys, run_matrix


# ---------------------------------------------------------------------------
# Exact solvers vs brute force
# ---------------------------------------------------------------------------

def _random_dist_matrix(n, seed):
    rng = random.Random(seed)
    coords = [(rng.randint(0, 100), rng.randint(0, 100)) for _ in range(n)]
    return [
        [round(((coords[i][0] - coords[j][0]) ** 2 + (coords[i][1] - coords[j][1]) ** 2) ** 0.5)
         for j in range(n)]
        for i in range(n)
    ]


@pytest.mark.parametrize("n", [3, 4, 5, 6, 7])
def test_held_karp_matches_brute_force(n):
    dist = _random_dist_matrix(n, seed=n)
    hk_cost, _ = held_karp(dist)
    bf_cost, _ = brute_force_tsp(dist)
    assert hk_cost == bf_cost


def test_knapsack_dp_matches_brute_force():
    rng = random.Random(42)
    weights = [rng.randint(1, 20) for _ in range(10)]
    values = [rng.randint(1, 50) for _ in range(10)]
    capacity = 40

    dp_value, dp_items = knapsack_dp(weights, values, capacity)
    bf_value, _ = brute_force_knapsack(weights, values, capacity)

    assert dp_value == bf_value
    assert sum(weights[i] for i in dp_items) <= capacity
    assert sum(values[i] for i in dp_items) == dp_value


# ---------------------------------------------------------------------------
# Feasibility
# ---------------------------------------------------------------------------

def _tiny_vrp_instance():
    dist = _random_dist_matrix(5, seed=1)
    return {"instance_id": 0, "size": 5, "seed": 1, "coords": [[0, 0]] * 5, "dist_matrix": dist,
            "optimal_cost": 0, "optimal_route": [0, 1, 2, 3, 4, 0]}


def test_vrp_rejects_missing_city():
    problem = VRPProblem(_tiny_vrp_instance())
    assert not problem.is_feasible([0, 1, 2, 3, 0])  # city 4 missing


def test_vrp_rejects_repeated_city():
    problem = VRPProblem(_tiny_vrp_instance())
    assert not problem.is_feasible([0, 1, 1, 2, 3, 4, 0])  # city 1 repeated


def test_vrp_accepts_complete_route():
    problem = VRPProblem(_tiny_vrp_instance())
    assert problem.is_feasible([0, 1, 2, 3, 4, 0])


def _tiny_knapsack_instance():
    return {"instance_id": 0, "size": 4, "seed": 1, "weights": [10, 20, 30, 15],
            "values": [60, 100, 120, 80], "capacity": 40,
            "optimal_value": 160, "optimal_items": [0, 1]}


def test_knapsack_rejects_over_capacity():
    problem = KnapsackProblem(_tiny_knapsack_instance())
    assert not problem.is_feasible([0, 1, 2, 3])  # weight 75 > capacity 40


def test_knapsack_accepts_within_capacity():
    problem = KnapsackProblem(_tiny_knapsack_instance())
    assert problem.is_feasible([0, 1])  # weight 30 <= 40


# ---------------------------------------------------------------------------
# Parsers recover from messy text, fail gracefully to a feasible sentinel
# ---------------------------------------------------------------------------

def test_vrp_parser_recovers_route_from_messy_text():
    problem = VRPProblem(_tiny_vrp_instance())
    text = "Well I think the best route is:\nRoute: [0, 2, 1, 3, 4, 0]\nThat's my answer."
    route = problem.parse_solution(text)
    assert problem.is_feasible(route)
    assert route == [0, 2, 1, 3, 4, 0]


def test_vrp_parser_falls_back_on_garbage():
    problem = VRPProblem(_tiny_vrp_instance())
    route = problem.parse_solution("I have no idea how to solve this.")
    assert problem.is_feasible(route)  # sentinel fallback is always feasible


def test_knapsack_parser_recovers_items_from_messy_text():
    problem = KnapsackProblem(_tiny_knapsack_instance())
    text = "After analysis, I choose items [0, 1, 1, 2] for the best value."
    items = problem.parse_solution(text)
    assert items == [0, 1, 2]  # de-duplicated, order preserved


def test_knapsack_parser_falls_back_on_garbage():
    problem = KnapsackProblem(_tiny_knapsack_instance())
    items = problem.parse_solution("no list here")
    assert items == []
    assert problem.is_feasible(items)  # empty selection is trivially feasible


# ---------------------------------------------------------------------------
# Call-count assertions, N=4, keep_ratio=0.5 (CLAUDE.md §5 table)
# ---------------------------------------------------------------------------

EXPECTED_CALLS_N4_KEEP_HALF = {
    "io": 1,
    "base_sge": 13,
    "sge_ap": 11,
    "kb_sge": 8,
    "kb_sge_ap": 6,
}


@pytest.mark.parametrize("method,expected", EXPECTED_CALLS_N4_KEEP_HALF.items())
def test_call_counts_match_formula(method, expected):
    assert registry.expected_calls(method, n=4, keep_ratio=0.5) == expected

    with open(config.KNOWLEDGE_BASE_PATH) as f:
        kb = json.load(f)
    problem = VRPProblem(_tiny_vrp_instance())
    llm_client = MockLLMClient()

    result = pipeline.run(method, problem, llm_client, kb, n=4, keep_ratio=0.5)

    assert result["calls"] == expected
    assert llm_client.count() == expected


# ---------------------------------------------------------------------------
# Checkpoint resume
# ---------------------------------------------------------------------------

def test_checkpoint_skips_completed_keys(tmp_path):
    csv_path = tmp_path / "results.csv"
    fieldnames = [
        "problem", "size", "method", "instance_id", "repeat", "N", "keep_ratio",
        "calls", "expected_calls", "prompt_tokens", "completion_tokens",
        "gap_pct", "objective", "optimal", "feasible",
        "methods_used", "hallucinated_count", "pruned_methods",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "problem": "vrp", "size": 5, "method": "io", "instance_id": 0, "repeat": 1,
            "N": 4, "keep_ratio": 0.5, "calls": 1, "expected_calls": 1,
            "prompt_tokens": 10, "completion_tokens": 10, "gap_pct": 0.0,
            "objective": 100, "optimal": 100, "feasible": True,
            "methods_used": "[]", "hallucinated_count": 0, "pruned_methods": "[]",
        })

    completed = load_completed_keys(str(csv_path))
    assert ("vrp", 5, "io", 0, 1) in completed
    assert ("vrp", 5, "io", 1, 1) not in completed


def test_run_matrix_resumes_without_rerunning(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    os.makedirs("results", exist_ok=True)

    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "core", "knowledge_base.json")) as f:
        kb = json.load(f)

    csv_path = "results/dryrun_resume_test.csv"
    llm_client = MockLLMClient()

    # First pass: config.NUM_INSTANCES instances x 1 repeat.
    run_matrix("vrp", [5], ["io"], repeats=1, llm_client=llm_client, kb=kb, csv_path=csv_path)

    with open(csv_path) as f:
        rows_after_first = list(csv.DictReader(f))
    assert len(rows_after_first) == config.NUM_INSTANCES

    # Second pass over the same matrix should add zero new rows (all skipped).
    run_matrix("vrp", [5], ["io"], repeats=1, llm_client=llm_client, kb=kb, csv_path=csv_path)

    with open(csv_path) as f:
        rows_after_second = list(csv.DictReader(f))
    assert len(rows_after_second) == config.NUM_INSTANCES
