"""Vehicle Routing Problem: city 0 = depot, single vehicle, visit all other
cities exactly once and return to depot. Objective = integer-Euclidean route
length (the distance matrix is precomputed once in core/instances.py and is
identical for every method, per invariant on shared distance matrices).
"""

import ast
import re

import config
from problems.base import Problem


class VRPProblem(Problem):
    kb_key = "VRP"

    def __init__(self, instance):
        self.instance = instance
        self.size = instance["size"]
        self.coords = instance["coords"]
        self.dist_matrix = instance["dist_matrix"]

    def create_prompt(self):
        lines = [
            "You are solving a Vehicle Routing Problem.",
            f"There are {self.size} locations indexed 0 to {self.size - 1}.",
            "Location 0 is the depot. The vehicle starts and ends at the depot.",
            "",
            "Coordinates (index: x, y):",
        ]
        for i, (x, y) in enumerate(self.coords):
            lines.append(f"  {i}: ({x}, {y})")

        lines.append("")
        lines.append("Distance matrix:")
        lines.append("     " + "  ".join(f"{j:4d}" for j in range(self.size)))
        for i in range(self.size):
            row = "  ".join(f"{self.dist_matrix[i][j]:4d}" for j in range(self.size))
            lines.append(f"  {i}: {row}")

        lines.append("")
        lines.append("Goal: find the shortest route starting and ending at the depot (0),")
        lines.append("visiting every other location exactly once.")

        return "\n".join(lines)

    def answer_format_hint(self):
        return "Return ONLY the route as a list, e.g. [0, 3, 1, 2, 4, 0]."

    def _fallback_route(self):
        return [0] + list(range(1, self.size)) + [0]

    def parse_solution(self, text):
        matches = re.findall(r"\[[\d,\s]+\]", text or "")
        for match in matches:
            try:
                route = ast.literal_eval(match)
                if not isinstance(route, list):
                    continue
                route = [int(x) for x in route]

                # detect and fix 1-indexed model output
                if route and max(route) >= self.size:
                    route = [max(0, x - 1) for x in route]

                route = [x for x in route if 0 <= x < self.size]
                if not route:
                    continue

                if route[0] != 0:
                    route = [0] + route
                if route[-1] != 0:
                    route = route + [0]

                if self.is_feasible(route):
                    return route
            except Exception:
                continue

        return self._fallback_route()

    def is_feasible(self, route):
        if not route or route[0] != 0 or route[-1] != 0:
            return False
        interior = route[1:-1]
        if sorted(interior) != list(range(1, self.size)):
            return False
        return True

    def raw_objective(self, route):
        if not route or len(route) < 2:
            return config.VRP_INFEASIBLE_COST
        total = 0
        for a, b in zip(route, route[1:]):
            if not (0 <= a < self.size and 0 <= b < self.size):
                return config.VRP_INFEASIBLE_COST
            total += self.dist_matrix[a][b]
        return total

    def score(self, route):
        if not self.is_feasible(route):
            return config.VRP_INFEASIBLE_COST
        return self.raw_objective(route)
