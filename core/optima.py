"""Exact solvers used to compute ground-truth optima. Never used by the LLM or
the pipeline's scoring function - only to compute the gap after a run completes.
"""

import math


def held_karp(dist_matrix):
    """Exact single-vehicle TSP: shortest route from depot (0) visiting every
    other node exactly once and returning to depot. O(n^2 * 2^n).

    Returns (cost, route) where route is [0, ..., 0].
    """
    n = len(dist_matrix)
    if n <= 1:
        return 0, [0, 0]

    customers = list(range(1, n))
    m = len(customers)
    full_mask = (1 << m) - 1

    dp = [[math.inf] * m for _ in range(1 << m)]
    parent = [[-1] * m for _ in range(1 << m)]

    for j in range(m):
        dp[1 << j][j] = dist_matrix[0][customers[j]]

    for mask in range(1 << m):
        for j in range(m):
            if not (mask & (1 << j)):
                continue
            cur = dp[mask][j]
            if cur == math.inf:
                continue
            for k in range(m):
                if mask & (1 << k):
                    continue
                new_mask = mask | (1 << k)
                new_cost = cur + dist_matrix[customers[j]][customers[k]]
                if new_cost < dp[new_mask][k]:
                    dp[new_mask][k] = new_cost
                    parent[new_mask][k] = j

    best_cost = math.inf
    best_j = -1
    for j in range(m):
        cost = dp[full_mask][j] + dist_matrix[customers[j]][0]
        if cost < best_cost:
            best_cost = cost
            best_j = j

    route_customers = []
    mask, j = full_mask, best_j
    while j != -1:
        route_customers.append(customers[j])
        pj = parent[mask][j]
        mask ^= (1 << j)
        j = pj
    route_customers.reverse()

    return int(best_cost), [0] + route_customers + [0]


def brute_force_tsp(dist_matrix):
    """Reference-only brute force for cross-checking Held-Karp on tiny instances."""
    from itertools import permutations

    n = len(dist_matrix)
    if n <= 1:
        return 0, [0, 0]
    customers = list(range(1, n))
    best_cost = math.inf
    best_route = None
    for perm in permutations(customers):
        route = [0] + list(perm) + [0]
        cost = sum(dist_matrix[route[i]][route[i + 1]] for i in range(len(route) - 1))
        if cost < best_cost:
            best_cost = cost
            best_route = route
    return int(best_cost), best_route


def knapsack_dp(weights, values, capacity):
    """Exact 0/1 knapsack via DP. O(n * capacity). Returns (best_value, chosen_indices)."""
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        w, v = weights[i - 1], values[i - 1]
        for c in range(capacity + 1):
            dp[i][c] = dp[i - 1][c]
            if w <= c:
                candidate = dp[i - 1][c - w] + v
                if candidate > dp[i][c]:
                    dp[i][c] = candidate

    chosen = []
    c = capacity
    for i in range(n, 0, -1):
        if dp[i][c] != dp[i - 1][c]:
            chosen.append(i - 1)
            c -= weights[i - 1]
    chosen.reverse()

    return dp[n][capacity], chosen


def brute_force_knapsack(weights, values, capacity):
    """Reference-only brute force for cross-checking the DP on tiny instances."""
    from itertools import combinations

    n = len(weights)
    best_value = 0
    best_items = []
    for r in range(n + 1):
        for combo in combinations(range(n), r):
            w = sum(weights[i] for i in combo)
            if w <= capacity:
                v = sum(values[i] for i in combo)
                if v > best_value:
                    best_value = v
                    best_items = list(combo)
    return best_value, best_items
