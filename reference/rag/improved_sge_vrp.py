"""
sge_vrp_fixed_instances.py
==========================
SGE (Simple Generative Evaluation) for VRP using fixed Easy / Medium / Hard instances.
"""

import re
import csv
import os
import time
import ast

import numpy as np
from groq import Groq
from dotenv import load_dotenv
from scipy.spatial import distance_matrix
from ortools.constraint_solver import pywrapcp, routing_enums_pb2


# ── CONFIG ─────────────────────────────────────────────────────────────

load_dotenv()
API_KEY = os.environ.get("GROQ_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "Missing GROQ_API_KEY. Set it in your environment or in a .env file."
    )
MODEL = "llama-3.3-70b-versatile"
SLEEP = 6
BACKOFF = 30

N_VALUES = [2, 3]   # number of trajectories to try per run
RUNS = 10
CSV_OUT = "results_vrp_sge_fixed.csv"

FIELDNAMES = [
    "system", "instance", "n", "run", "calls",
    "best_cost", "best_route", "optimal_cost", "gap"
]


# ── FIXED CITY INSTANCES ───────────────────────────────────────────────

EASY_CITIES = [
    (25, 30),   # 0 depot
    (80, 10),
    (15, 85),
    (60, 55),
    (40, 20),
    (90, 70),
]

MEDIUM_CITIES = [
    (50, 50),   # 0 depot
    (10, 10),
    (90, 20),
    (20, 80),
    (85, 85),
    (60, 15),
    (15, 60),
    (70, 40),
    (40, 90),
]

HARD_CITIES = [
    (50, 50),   # 0 depot (center)
    (5,  5),   # corners
    (95,  5),
    (5, 95),
    (95, 95),
    (30, 40),   # tricky cluster 1
    (35, 45),
    (70, 60),   # tricky cluster 2
    (75, 65),
    (52, 20),   # misleading near-center
]

INSTANCES = {
    "easy":   EASY_CITIES,
    "medium": MEDIUM_CITIES,
    "hard":   HARD_CITIES,
}


# ── VRP PROBLEM CLASS ──────────────────────────────────────────────────

class VRPProblem:
    """
    VRP problem built from a fixed list of (x, y) city coordinates.
    City 0 is always the depot.
    """

    def __init__(self, coords):
        self.coords = list(coords)
        self.num_cities = len(self.coords)

        coords_np = np.array(self.coords, dtype=float)
        dm = distance_matrix(coords_np, coords_np)
        self.dist_mat = dm.astype(int).tolist()

        self._optimal_cost = None
        self._optimal_route = None

    # ── COST ──────────────────────────────────────────────────────────

    def compute_cost(self, route):
        if not route or len(route) < 2:
            return 99999

        interior = [c for c in route if c != 0]
        if set(interior) != set(range(1, self.num_cities)):
            return 99999

        cost = 0
        for i in range(1, len(route)):
            a, b = route[i - 1], route[i]
            if a >= self.num_cities or b >= self.num_cities:
                return 99999
            cost += self.dist_mat[a][b]

        return cost

    # ── OPTIMAL (OR-TOOLS) ────────────────────────────────────────────

    def solve_optimal(self):
        if self._optimal_cost is not None:
            return self._optimal_route, self._optimal_cost

        manager = pywrapcp.RoutingIndexManager(self.num_cities, 1, 0)
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            return self.dist_mat[manager.IndexToNode(from_index)][
                manager.IndexToNode(to_index)
            ]

        transit = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit)

        params = pywrapcp.DefaultRoutingSearchParameters()
        params.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )

        solution = routing.SolveWithParameters(params)

        if solution:
            index = routing.Start(0)
            route = []
            while not routing.IsEnd(index):
                route.append(manager.IndexToNode(index))
                index = solution.Value(routing.NextVar(index))
            route.append(manager.IndexToNode(index))

            cost = self.compute_cost(route)
            self._optimal_route = route
            self._optimal_cost = cost
            return route, cost

        return None, None

    def compute_gap(self, route):
        _, optimal_cost = self.solve_optimal()
        if optimal_cost is None or optimal_cost <= 0:
            return -1
        model_cost = self.compute_cost(route)
        return max(0, (model_cost - optimal_cost) / optimal_cost * 100)

    # ── PROMPT ────────────────────────────────────────────────────────

    def create_prompt(self):
        lines = [
            "You are solving a Vehicle Routing Problem.",
            f"There are {self.num_cities} locations indexed 0 to {self.num_cities - 1}.",
            "Location 0 is the depot.",
            "Return a single route that starts and ends at 0 and visits every other location exactly once.",
            "",
            "Coordinates:",
        ]
        for i, (x, y) in enumerate(self.coords):
            lines.append(f"  {i}: ({x}, {y})")

        lines.append("")
        lines.append("Distance matrix (integer Euclidean):")
        for i in range(self.num_cities):
            row = " ".join(
                f"{self.dist_mat[i][j]:4d}" for j in range(self.num_cities))
            lines.append(f"  {i}: {row}")

        lines.append("")
        lines.append("Return ONLY a route like [0, 2, 1, 3, 4, 0]")
        return "\n".join(lines)

    # ── PARSE LLM OUTPUT ──────────────────────────────────────────────

    def parse_solution(self, text):
        matches = re.findall(r"\[[\d,\s]+\]", text)

        best_route = None
        best_cost = float("inf")

        for m in matches:
            try:
                route = ast.literal_eval(m)
                if not isinstance(route, list):
                    continue

                route = [int(x)
                         for x in route if 0 <= int(x) < self.num_cities]
                if not route:
                    continue

                if route[0] != 0:
                    route = [0] + route
                if route[-1] != 0:
                    route.append(0)

                cost = self.compute_cost(route)
                if cost < best_cost:
                    best_cost = cost
                    best_route = route
            except Exception:
                continue

        if best_route:
            return best_route

        # Safe fallback: trivial tour
        return [0] + list(range(1, self.num_cities)) + [0]


# ── LLM CALL ──────────────────────────────────────────────────────────

client = Groq(api_key=API_KEY)
call_count = 0


def llm(prompt, phase):
    global call_count

    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}]
            )
            out = resp.choices[0].message.content
            call_count += 1
            time.sleep(SLEEP)

            print(f"\n{'='*60}")
            print(f"{phase} (call #{call_count})")
            print(f"{'='*60}")
            print(out)

            return out

        except Exception as e:
            if "429" in str(e):
                wait = BACKOFF * (2 ** attempt)
                print(f"Rate limit → wait {wait}s")
                time.sleep(wait)
            else:
                print(e)
                return ""

    return ""


# ── METHOD PARSER ─────────────────────────────────────────────────────

def parse_methods(text, n_trajectories):
    methods = []
    for line in text.splitlines():
        clean = re.sub(r"^[\s\*\-\d\.\)]+", "", line).strip()
        if len(clean) >= 3:
            methods.append(clean)
        if len(methods) == n_trajectories:
            break
    return methods


# ── SINGLE SGE RUN ────────────────────────────────────────────────────

def run_single(difficulty: str, n_trajectories: int):
    """
    Run one SGE trial on the specified fixed instance.
    Returns (best_cost, best_route, opt_cost, gap) or None on failure.
    """
    global call_count
    call_count = 0

    coords = INSTANCES[difficulty]
    problem = VRPProblem(coords)
    QUESTION = problem.create_prompt()

    print(
        f"\n=== INSTANCE: {difficulty.upper()} ({problem.num_cities} cities) ===")
    print(QUESTION)

    # ── PHASE 1 — Exploration ─────────────────────────────────────────
    raw = llm(
        QUESTION + "\n\n"
        f"List {n_trajectories} simple or intuitive ways to approach this routing problem. "
        "They do not have to be optimal or well-known.",
        "PHASE 1 — Exploration"
    )

    methods = parse_methods(raw, n_trajectories)
    print(f"\nParsed methods: {methods}")

    if not methods:
        print("Phase 1 failed — no methods parsed.")
        return None

    best_route = None
    best_cost = 99999

    # ── PHASES 2–4 per trajectory ─────────────────────────────────────
    for idx, method in enumerate(methods):
        n = idx + 1

        print(f"\n\n{'#'*60}")
        print(f"TRAJECTORY {n}/{len(methods)}: {method}")
        print(f"{'#'*60}")

        # PHASE 2 — Decomposition
        steps = llm(
            f"{QUESTION}\n\nMethod: {method}\n"
            "List only the steps to apply this method to solve the VRP. "
            "One step per line. No code, no explanation.",
            f"PHASE 2 — Decomposition [T{n}]"
        )

        # PHASE 3 — Action
        action = llm(
            f"{QUESTION}\n\nMethod: {method}\nSteps:\n{steps}\n\n"
            "Apply these steps manually to construct a valid route. "
            "You must visit every location exactly once and return to depot (0). "
            "A quick and intuitive solution is fine — it does not need to be optimal. "
            "You can make decisions greedily without checking all options."
            "Do NOT write code. Show reasoning step by step. "
            "End with ONLY a Python list representing the route "
            "like [0, 2, 1, 3, 4, 0].",
            f"PHASE 3 — Action [T{n}]"
        )

        route = problem.parse_solution(action)
        cost = problem.compute_cost(route)
        print(f"\nRoute: {route}  Cost: {cost}")

        # PHASE 4 — Refinement
        refined = llm(
            f"{QUESTION}\n\nCurrent route:\n{route}\n\n"
            "Check if this route can be improved by reordering cities. "
            "Try swaps or local improvements to reduce total distance. "
            "Do NOT write code. Show reasoning. "
            "End with ONLY the final improved route as a Python list.",
            f"PHASE 4 — Refinement [T{n}]"
        )

        route = problem.parse_solution(refined)
        cost = problem.compute_cost(route)
        print(f"\nRefined Route: {route}  Refined Cost: {cost}")

        if cost < best_cost:
            best_cost = cost
            best_route = route

    opt_route, opt_cost = problem.solve_optimal()
    gap = problem.compute_gap(best_route)

    print("\n=== FINAL RESULT ===")
    print(f"Best route   : {best_route}")
    print(f"Best cost    : {best_cost}")
    print(f"Optimal cost : {opt_cost}")
    print(f"Gap (%%)      : {gap:.2f}")

    return best_cost, best_route, opt_cost, gap


# ── CHECKPOINTING ─────────────────────────────────────────────────────

def load_completed_runs(path):
    completed = set()
    if not os.path.exists(path):
        return completed
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            completed.add((row["instance"], int(row["n"]), int(row["run"])))
    return completed


def open_csv(path):
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    f = open(path, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    if not exists:
        writer.writeheader()
    return f, writer


# ── MAIN LOOP ─────────────────────────────────────────────────────────

def main():
    completed = load_completed_runs(CSV_OUT)
    total = len(N_VALUES) * len(INSTANCES) * RUNS
    print(f"Checkpoint: {len(completed)}/{total} runs already done.")

    f, writer = open_csv(CSV_OUT)

    try:
        for n in N_VALUES:
            for difficulty in ["easy", "medium", "hard"]:
                for run_id in range(1, RUNS + 1):
                    key = (difficulty, n, run_id)

                    if key in completed:
                        print(f"Skip {key}")
                        continue

                    print(f"\n{'*'*60}")
                    print(
                        f"START  difficulty={difficulty}  n={n}  run={run_id}")
                    print(f"{'*'*60}")

                    result = run_single(difficulty, n)

                    if result is None:
                        print(f"✗ run_single returned None — skipping {key}")
                        continue

                    best_cost, best_route, opt_cost, gap = result

                    writer.writerow({
                        "system":       "SGE",
                        "instance":     difficulty,
                        "n":            n,
                        "run":          run_id,
                        "calls":        call_count,
                        "best_cost":    best_cost,
                        "best_route":   str(best_route),
                        "optimal_cost": opt_cost,
                        "gap":          f"{gap:.4f}",
                    })
                    f.flush()
                    completed.add(key)

                    print(f"✓ Saved {key}  cost={best_cost}  gap={gap:.2f}%")

    except KeyboardInterrupt:
        print("\nStopped by user. Progress saved — run again to resume.")

    finally:
        f.close()

    print(f"\nAll results saved to {CSV_OUT}")


if __name__ == "__main__":
    main()
