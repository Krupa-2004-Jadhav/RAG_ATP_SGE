"""Deterministic, rule-based KB method selection (CLAUDE.md §7). This is
plain lookup keyed on instance features - NOT retrieval-augmented generation
(no embeddings, no vector search, no semantic similarity). Zero LLM calls.

Rule rationale:
  - VRP, small (<= 8 customer-scale locations): a fast constructive method
    (Nearest Neighbor) paired with a local-search improver (2-Opt) is enough
    to be competitive when the search space is small.
  - VRP, larger: a construction heuristic that reasons about savings
    (Clarke-Wright) plus a metaheuristic (Simulated Annealing) that can escape
    local optima matters more as the space grows.
  - Knapsack, tight capacity (capacity / total_weight low): item selection is
    highly constrained, so exact methods (DP, Branch and Bound) that reason
    about the constraint directly are picked over a greedy ratio heuristic.
  - Knapsack, loose capacity: greedy-by-ratio is usually near-optimal and
    cheap, paired with DP as the exact fallback.
Every rule always returns one constructive/greedy method and one
improvement/exact method, per CLAUDE.md.

If N (the trajectory count) exceeds the rule's picks, the remaining slots are
filled deterministically by walking the KB's method list for that problem in
its stored order, skipping names already picked - no randomness.
"""

VRP_SMALL_SIZE_THRESHOLD = 8
KNAPSACK_TIGHT_RATIO_THRESHOLD = 0.5

VRP_SMALL_RULE = ["Nearest Neighbor", "2-Opt"]
VRP_LARGE_RULE = ["Clarke-Wright Savings Algorithm", "Simulated Annealing"]

KNAPSACK_TIGHT_RULE = ["Dynamic Programming", "Branch and Bound"]
KNAPSACK_LOOSE_RULE = ["Greedy Algorithm", "Dynamic Programming"]


def _rule_names(problem_key, instance):
    if problem_key == "VRP":
        if instance["size"] <= VRP_SMALL_SIZE_THRESHOLD:
            return list(VRP_SMALL_RULE)
        return list(VRP_LARGE_RULE)

    if problem_key == "Knapsack":
        total_weight = sum(instance["weights"])
        ratio = instance["capacity"] / total_weight if total_weight else 0
        if ratio < KNAPSACK_TIGHT_RATIO_THRESHOLD:
            return list(KNAPSACK_TIGHT_RULE)
        return list(KNAPSACK_LOOSE_RULE)

    raise ValueError(f"No retrieval rule for problem {problem_key!r}")


def _find_by_name(pool, name):
    for entry in pool:
        if entry["name"].lower() == name.lower():
            return entry
    return None


def select_methods(problem_key, instance, kb, n):
    """Returns up to n [{name, steps}] entries straight from the KB, chosen by
    instance features. Zero LLM calls.
    """
    pool = kb[problem_key]["methods"]
    names = _rule_names(problem_key, instance)

    selected = []
    for name in names:
        entry = _find_by_name(pool, name)
        if entry is not None and entry not in selected:
            selected.append(entry)

    if len(selected) < n:
        for entry in pool:
            if len(selected) >= n:
                break
            if entry not in selected:
                selected.append(entry)

    return [{"name": e["name"], "steps": list(e["steps"])} for e in selected[:n]]


def known_method_names(problem_key, kb):
    """KB vocabulary for a problem, used for hallucination detection in base_sge/sge_ap."""
    return {e["name"].lower() for e in kb[problem_key]["methods"]}
