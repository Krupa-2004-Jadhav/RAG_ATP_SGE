"""0/1 Knapsack: maximize value subject to weight <= capacity. Items are
0-indexed. Internal score is negative value (lower-is-better convention) so
the pipeline's argmin integration works identically across problems.
"""

import ast
import re

import config
from problems.base import Problem


class KnapsackProblem(Problem):
    kb_key = "Knapsack"

    def __init__(self, instance):
        self.instance = instance
        self.size = instance["size"]
        self.weights = instance["weights"]
        self.values = instance["values"]
        self.capacity = instance["capacity"]

    def create_prompt(self):
        lines = [
            "You are solving a 0/1 Knapsack problem.",
            f"There are {self.size} items, indexed 0 to {self.size - 1}.",
            f"Knapsack capacity: {self.capacity}",
            "",
            "Items (index: weight, value):",
        ]
        for i in range(self.size):
            lines.append(f"  {i}: weight={self.weights[i]}, value={self.values[i]}")

        lines.append("")
        lines.append("Goal: choose a subset of items (each used at most once) that")
        lines.append("maximizes total value without the total weight exceeding capacity.")

        return "\n".join(lines)

    def answer_format_hint(self):
        return "Return ONLY the list of chosen item indices (0-indexed), e.g. [0, 2, 5]."

    def _clean_items(self, items):
        seen = []
        for x in items:
            if isinstance(x, int) and 0 <= x < self.size and x not in seen:
                seen.append(x)
        return seen

    def parse_solution(self, text):
        matches = re.findall(r"\[[\d,\s]*\]", text or "")
        for match in reversed(matches):
            try:
                items = ast.literal_eval(match)
                if not isinstance(items, list):
                    continue
                items = self._clean_items([int(x) for x in items])
                return items
            except Exception:
                continue
        return []

    def is_feasible(self, items):
        if items is None:
            return False
        total_weight = sum(self.weights[i] for i in items if 0 <= i < self.size)
        return total_weight <= self.capacity

    def raw_objective(self, items):
        if not items:
            return 0
        return sum(self.values[i] for i in items if 0 <= i < self.size)

    def score(self, items):
        if not self.is_feasible(items):
            return config.KNAPSACK_INFEASIBLE_SCORE
        return -self.raw_objective(items)
