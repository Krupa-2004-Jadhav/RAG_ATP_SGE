"""
improved_rag_sge_vrp.py
=======================
RAG-SGE for Vehicle Routing Problem (VRP) — improved experiment version.

Settings:
  - N = 2 (fixed, matches SGE experiment)
  - 3 fixed instances (easy / medium / hard), each run 10 times = 30 total runs
  - Checkpointing: swap API_KEY and re-run to resume automatically
  - Phase 5: automated solution picker (no manual reading)
  - Results saved to results_rag_sge_vrp.csv after every run

LLM calls per run: 2 × N = 4  (vs SGE's 7)

Each instance is FIXED (hardcoded cities + distance matrix).
City 0 is always the depot.
"""

import re
import csv
import json
import os
import time
import math
from itertools import permutations
from groq import Groq
from dotenv import load_dotenv

# ── Config ────────────────────────────────────────────────────────────────────

load_dotenv()
API_KEY = os.environ.get("GROQ_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "Missing GROQ_API_KEY. Set it in your environment or in a .env file."
    )
MODEL = "llama-3.3-70b-versatile"
SLEEP = 6
BACKOFF = 30
N = 2
RUNS = 10          # repeat each fixed instance this many times
CSV_OUT = "results_rag_sge_vrp.csv"
KB_PATH = "knowledge_base.json"

# ── Fixed city coordinates ────────────────────────────────────────────────────
# City 0 = depot in every instance.

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

    # corners (create long edges)
    (5, 5),
    (95, 5),
    (5, 95),
    (95, 95),

    # tricky cluster 1
    (30, 40),
    (35, 45),

    # tricky cluster 2
    (70, 60),
    (75, 65),

    # misleading near-center point
    (52, 20),
]

# ── Build distance matrix from city list ──────────────────────────────────────


def build_dist_matrix(cities: list) -> list:
    n = len(cities)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                dx = cities[i][0] - cities[j][0]
                dy = cities[i][1] - cities[j][1]
                dist[i][j] = round(math.sqrt(dx*dx + dy*dy), 2)
    return dist


# ── Build INSTANCES dict ──────────────────────────────────────────────────────

def _make_instance(cities: list, label: str) -> dict:
    dist = build_dist_matrix(cities)
    n = len(cities)

    coord_lines = "\n".join(
        f"  City {i} {'(depot)' if i == 0 else '       '}: ({cities[i][0]}, {cities[i][1]})"
        for i in range(n)
    )
    header = "      " + "".join(f"  C{j:<3}" for j in range(n))
    rows = [f"C{i:<3}  " + "".join(f"{dist[i][j]:6.1f}" for j in range(n))
            for i in range(n)]
    dist_str = header + "\n" + "\n".join(rows)

    description = (
        f"VRP instance ({label}): {n} cities, 1 vehicle.\n"
        f"City 0 is the depot. Visit all other cities exactly once and return to depot.\n"
        f"Minimize total travel distance.\n\n"
        f"City coordinates:\n{coord_lines}\n\n"
        f"Distance matrix (Euclidean, rounded to 2 dp):\n{dist_str}"
    )
    return {"cities": cities, "dist_matrix": dist, "description": description}


INSTANCES = {
    "easy":   _make_instance(EASY_CITIES,   "easy,   5 cities"),
    "medium": _make_instance(MEDIUM_CITIES, "medium, 8 cities"),
    "hard":   _make_instance(HARD_CITIES,   "hard,  10 cities"),
}


# ── Brute-force optimal (feasible for up to 10 cities) ───────────────────────

def brute_force_optimal(dist_matrix: list) -> tuple:
    """Return (best_distance, best_route). City 0 is depot."""
    n = len(dist_matrix)
    customers = list(range(1, n))
    best_dist = float("inf")
    best_route = None
    for perm in permutations(customers):
        route = [0] + list(perm) + [0]
        d = sum(dist_matrix[route[i]][route[i+1]] for i in range(len(route)-1))
        if d < best_dist:
            best_dist = d
            best_route = route
    return round(best_dist, 2), best_route


# Pre-compute optima once at startup
OPTIMA = {
    name: brute_force_optimal(inst["dist_matrix"])
    for name, inst in INSTANCES.items()
}


# ── CSV helpers ───────────────────────────────────────────────────────────────

FIELDNAMES = [
    "system", "instance", "run", "n",
    "calls", "best_distance", "best_route",
    "best_method", "feasible",
    "optimal_distance", "gap_pct",
    "hallucination_count", "hallucinated_methods",
]


def load_completed_runs(csv_path: str) -> set:
    completed = set()
    if not os.path.exists(csv_path):
        return completed
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                completed.add(
                    (row["instance"], int(row["run"]), int(row["n"])))
            except (KeyError, ValueError):
                pass
    return completed


def open_csv_for_append(csv_path: str):
    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    f = open(csv_path, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    if not file_exists:
        writer.writeheader()
    return f, writer


# ── Rate-limit sentinel ───────────────────────────────────────────────────────

class RateLimitExhausted(Exception):
    pass


# ── Knowledge-base helpers ────────────────────────────────────────────────────

def load_kb(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def retrieve_methods(kb: dict, problem: str, n: int) -> list:
    methods = kb.get(problem, {}).get("methods", [])[:n]
    print(f"\n{'='*55}")
    print(f"  PHASE 1 (KB lookup) — retrieved {len(methods)} methods")
    print(f"{'='*55}")
    for m in methods:
        print(f"  • {m['name']}")
    return methods


def retrieve_steps(method: dict) -> str:
    steps = method.get("steps", [])
    steps_text = (
        "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))
        if isinstance(steps, list) else str(steps)
    )
    print(f"\n{'='*55}")
    print(f"  PHASE 2 (KB lookup) — steps for: {method['name']}")
    print(f"{'='*55}")
    print(steps_text)
    return steps_text


# ── LLM call ─────────────────────────────────────────────────────────────────

call_count = 0


def llm(client, prompt: str, phase: str) -> str:
    global call_count
    for attempt in range(5):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}]
            )
            output = resp.choices[0].message.content
            call_count += 1
            time.sleep(SLEEP)
            print(f"\n{'='*55}\n  {phase}  (call #{call_count})\n{'='*55}")
            print(output)
            return output
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                if attempt < 4:
                    wait = BACKOFF * (2 ** attempt)
                    print(
                        f"  Rate limited. Waiting {wait}s... (attempt {attempt+1}/5)")
                    time.sleep(wait)
                else:
                    print("\n  !! Rate limit exhausted after 5 attempts.")
                    raise RateLimitExhausted()
            else:
                print(f"  Error: {e}")
                return ""
    return ""


# ── Route parsing & evaluation ────────────────────────────────────────────────

def extract_route(text: str) -> list:
    matches = re.findall(r"\[[\d,\s]+\]", text)
    if not matches:
        return []
    for m in reversed(matches):   # prefer last match (usually final answer)
        try:
            route = [int(x) for x in re.findall(r"\d+", m)]
            if len(route) >= 2:
                return route
        except Exception:
            continue
    return []


def evaluate_route(route: list, dist_matrix: list) -> tuple:
    """Returns (total_distance, feasible)."""
    n = len(dist_matrix)
    customers = set(range(1, n))

    if not route or route[0] != 0 or route[-1] != 0:
        return float("inf"), False

    visited = route[1:-1]
    if sorted(visited) != sorted(customers):
        return float("inf"), False
    if len(visited) != len(set(visited)):
        return float("inf"), False

    total = sum(dist_matrix[route[i]][route[i+1]] for i in range(len(route)-1))
    return round(total, 2), True


def phase5_pick_best(trajectories: list, dist_matrix: list) -> dict:
    best = {"distance": float("inf"), "route": [],
            "method": "none", "feasible": False}
    for traj in trajectories:
        route = extract_route(traj["phase4_output"])
        if not route:
            continue
        dist, feasible = evaluate_route(route, dist_matrix)
        if feasible and dist < best["distance"]:
            best = {"distance": dist, "route": route,
                    "method": traj["method"], "feasible": True}
    return best


# ── Single RAG-SGE run ────────────────────────────────────────────────────────

def run_rag_sge(client, kb: dict, inst_name: str, inst: dict, run_id: int) -> dict:
    global call_count
    call_count = 0

    dist_matrix = inst["dist_matrix"]
    optimal_dist, opt_route = OPTIMA[inst_name]

    full_question = (
        f"Solve this Vehicle Routing Problem (VRP).\n{inst['description']}\n\n"
        "Rules:\n"
        "  1. Start and end at City 0 (depot).\n"
        "  2. Visit every other city exactly once.\n"
        "  3. Minimize total travel distance.\n"
        "  4. Do NOT write code — reason step by step.\n"
        "  5. End your answer with a Python list of city indices, e.g. [0, 2, 1, 3, 0]."
    )

    print(f"\n\n{'#'*60}")
    print(
        f"  RAG-SGE VRP | instance={inst_name} | run={run_id}/{RUNS} | N={N}")
    print(f"  Optimal distance : {optimal_dist}  Route: {opt_route}")
    print(f"{'#'*60}")

    # PHASE 1 — KB lookup (0 LLM calls)
    methods = retrieve_methods(kb, problem="VRP", n=N)
    if not methods:
        return _empty_result(inst_name, run_id, optimal_dist)

    trajectories = []
    for idx, method in enumerate(methods):
        t = idx + 1
        method_name = method["name"]
        print(f"\n\n{'='*55}\n  TRAJECTORY {t}/{N}: {method_name}\n{'='*55}")

        # PHASE 2 — KB lookup (0 LLM calls)
        steps = retrieve_steps(method)

        # PHASE 3 — LLM applies steps to the fixed instance
        action = llm(client,
                     f"{full_question}\n\nMethod: {method_name}\nSteps:\n{steps}\n\n"
                     "Apply these steps to solve the problem. Show your working clearly. "
                     "End with a Python list of city indices representing the full route, "
                     "e.g. [0, 3, 1, 2, 0].",
                     f"PHASE 3 — Action [T{t}]"
                     )

        # PHASE 4 — LLM refines
        phase4 = llm(client,
                     f"{full_question}\n\nCurrent route:\n{action}\n\n"
                     "Check for errors:\n"
                     "  • Does the route start and end at City 0?\n"
                     "  • Are all cities visited exactly once?\n"
                     "  • Can any 2-opt swap reduce the total distance?\n"
                     "Show all calculations. "
                     "End with your final improved Python list of city indices.",
                     f"PHASE 4 — Refinement [T{t}]"
                     )

        trajectories.append({"method": method_name, "phase4_output": phase4})

    # PHASE 5 — auto-pick best feasible solution
    best = phase5_pick_best(trajectories, dist_matrix)
    gap_pct = None
    if best["feasible"] and optimal_dist > 0:
        gap_pct = round(
            ((best["distance"] - optimal_dist) / optimal_dist) * 100, 2)

    print(f"\n  PHASE 5 — Best: route={best['route']}, dist={best['distance']}, "
          f"method={best['method']}, feasible={best['feasible']}, gap={gap_pct}%")
    print(f"\n{'='*55}")
    print(f"  LLM calls : {call_count}  |  KB lookups: {1 + N}")
    print(f"  SGE would : {1 + 3*N} calls  |  Saved: {1 + 3*N - call_count}")
    print(
        f"  Optimal   : {optimal_dist}  |  Found: {best['distance']}  |  Gap: {gap_pct}%")
    print(f"{'='*55}")

    return {
        "system":               "RAG-SGE",
        "instance":             inst_name,
        "run":                  run_id,
        "n":                    N,
        "calls":                call_count,
        "best_distance":        best["distance"] if best["feasible"] else None,
        "best_route":           str(best["route"]),
        "best_method":          best["method"],
        "feasible":             best["feasible"],
        "optimal_distance":     optimal_dist,
        "gap_pct":              gap_pct,
        "hallucination_count":  0,   # always 0 — methods come from KB
        "hallucinated_methods": "",
    }


def _empty_result(inst_name: str, run_id: int, optimal_dist: float) -> dict:
    return {
        "system": "RAG-SGE", "instance": inst_name, "run": run_id, "n": N,
        "calls": 0, "best_distance": None, "best_route": "[]",
        "best_method": "none", "feasible": False,
        "optimal_distance": optimal_dist, "gap_pct": None,
        "hallucination_count": 0, "hallucinated_methods": "",
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    try:
        kb = load_kb(KB_PATH)
    except FileNotFoundError:
        print(f"ERROR: '{KB_PATH}' not found.")
        return

    kb_methods = kb.get("VRP", {}).get("methods", [])
    if len(kb_methods) < N:
        print(f"ERROR: KB has only {len(kb_methods)} VRP methods, need {N}.")
        return
    print(
        f"  KB loaded — {len(kb_methods)} VRP methods available, using first {N}.")

    # Print pre-computed optima
    print("\n  Pre-computed optimal distances (brute force):")
    for name, (opt_dist, opt_route) in OPTIMA.items():
        print(f"    {name:6s}: {opt_dist}  {opt_route}")

    total = len(INSTANCES) * RUNS      # 3 × 10 = 30
    completed = load_completed_runs(CSV_OUT)
    remaining = total - len(completed)
    print(
        f"\n  Checkpoint: {len(completed)}/{total} runs done, {remaining} remaining.")
    if not remaining:
        print("  Nothing to do — all runs complete!")
        return

    client = Groq(api_key=API_KEY)
    f, writer = open_csv_for_append(CSV_OUT)
    done = len(completed)

    try:
        for inst_name, inst in INSTANCES.items():
            for run_id in range(1, RUNS + 1):
                key = (inst_name, run_id, N)
                if key in completed:
                    print(f"  → Skip (done): {inst_name} run={run_id}")
                    continue

                result = run_rag_sge(client, kb, inst_name, inst, run_id)
                writer.writerow(result)
                f.flush()
                completed.add(key)
                done += 1

                print(
                    f"\n  ✓ [{done}/{total}] {inst_name} | run={run_id} | "
                    f"dist={result['best_distance']} | "
                    f"optimal={result['optimal_distance']} | "
                    f"gap={result['gap_pct']}% | calls={result['calls']}"
                )

    except RateLimitExhausted:
        print(f"\n\n{'!'*60}")
        print(f"  RATE LIMIT EXHAUSTED — stopped cleanly.")
        print(f"  Progress : {done}/{total} runs saved to {CSV_OUT}")
        print(f"  Remaining: {total - done} runs")
        print(f"\n  TO RESUME:")
        print(f"  1. Replace API_KEY at the top of this file with a fresh key.")
        print(f"  2. Run again — completed runs are skipped automatically.")
        print(f"{'!'*60}")

    except KeyboardInterrupt:
        print(f"\n\n  Stopped by user. {done}/{total} runs saved.")
        print(f"  Re-run to continue from where you left off.")

    finally:
        f.close()

    print(f"\n  All done. Results saved to {CSV_OUT}")


if __name__ == "__main__":
    main()
