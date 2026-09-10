"""Fixed-instance generation. Generate once from seeds, save to JSON, and every
method loads the same file (CLAUDE.md invariant #1). Exact optima (held_karp /
knapsack_dp) are computed once here at generation time and embedded in the
JSON, never recomputed per run (invariant #3: no optimum leakage to the LLM -
the optimum is only ever read back to compute the gap after a run).
"""

import json
import os

import numpy as np

import config
from core.optima import held_karp, knapsack_dp


def _vrp_instance(size, seed):
    rng = np.random.RandomState(seed)
    coords = rng.randint(0, 100, size=(size, 2))

    dist_matrix = [[0] * size for _ in range(size)]
    for i in range(size):
        for j in range(size):
            if i != j:
                dx = int(coords[i][0]) - int(coords[j][0])
                dy = int(coords[i][1]) - int(coords[j][1])
                dist_matrix[i][j] = round((dx * dx + dy * dy) ** 0.5)

    optimal_cost, optimal_route = held_karp(dist_matrix)

    return {
        "instance_id": None,
        "size": size,
        "seed": seed,
        "coords": coords.tolist(),
        "dist_matrix": dist_matrix,
        "optimal_cost": optimal_cost,
        "optimal_route": optimal_route,
    }


def _knapsack_instance(size, seed):
    rng = np.random.RandomState(seed)
    weights = rng.randint(5, 50, size=size).tolist()
    values = rng.randint(10, 200, size=size).tolist()
    ratio = float(rng.uniform(0.3, 0.7))
    capacity = max(1, int(ratio * sum(weights)))

    optimal_value, optimal_items = knapsack_dp(weights, values, capacity)

    return {
        "instance_id": None,
        "size": size,
        "seed": seed,
        "weights": weights,
        "values": values,
        "capacity": capacity,
        "optimal_value": optimal_value,
        "optimal_items": optimal_items,
    }


_GENERATORS = {"vrp": _vrp_instance, "knapsack": _knapsack_instance}


def generate_instances(problem, size, num_instances=config.NUM_INSTANCES):
    generator = _GENERATORS[problem]
    instances = []
    for instance_id in range(num_instances):
        seed = config.SEED_BASE + instance_id
        inst = generator(size, seed)
        inst["instance_id"] = instance_id
        instances.append(inst)
    return instances


def instances_path(problem, size):
    return config.INSTANCES_PATH_TEMPLATE.format(problem=problem, size=size)


def load_or_generate(problem, size):
    """Load the fixed instance set for (problem, size), generating and saving
    it once if the file doesn't exist yet. Every caller (every method, every
    repeat) gets the exact same instances.
    """
    path = instances_path(problem, size)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)

    instances = generate_instances(problem, size)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(instances, f, indent=2)
    return instances
