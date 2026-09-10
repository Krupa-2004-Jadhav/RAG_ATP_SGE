"""
improved_sge_knapsack.py
========================
Stronger SGE experiment for Knapsack only.

Changes vs original sge_knapsack.py:
  1. 10 runs per configuration
  2. 3 knapsack instances (Easy / Medium / Hard)
  3. Automated Phase 5 — parses & picks best feasible solution
  4. Tests N=2 and N=3
  5. Tracks hallucination rate in Phase 1
  6. Saves all results to results_sge.csv
  7. CHECKPOINTING — resumes from where it left off if interrupted
     Just swap API_KEY and re-run. Completed runs are never repeated.

Total LLM calls per run:
  N=2 → 1 + (3×2) = 7
  N=3 → 1 + (3×3) = 10
"""

import re
import csv
import os
import time
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
N_VALUES = [2, 3]
RUNS = 10
CSV_OUT = "results_sge.csv"

# ── Knapsack instances ────────────────────────────────────────────────────────

INSTANCES = {
    "easy": {
        "capacity": 100,
        "items": [
            (10, 60), (20, 100), (30, 120), (15, 80), (25, 90),
            (35, 150), (12, 70), (22, 110), (18, 95), (40, 200),
        ],
        "description": (
            "Capacity = 100. Items 1-10: "
            "Item1(w=10,v=60), Item2(w=20,v=100), Item3(w=30,v=120), "
            "Item4(w=15,v=80), Item5(w=25,v=90), Item6(w=35,v=150), "
            "Item7(w=12,v=70), Item8(w=22,v=110), Item9(w=18,v=95), "
            "Item10(w=40,v=200)."
        ),
    },
    "medium": {
        "capacity": 150,
        "items": [
            (10, 60), (20, 100), (30, 120), (15, 80), (25, 90),
            (35, 150), (12, 70), (22, 110), (18, 95), (40, 200),
            (28, 130), (33, 160), (9, 45), (17, 85), (50, 220),
        ],
        "description": (
            "Capacity = 150. Items 1-15: "
            "Item1(w=10,v=60), Item2(w=20,v=100), Item3(w=30,v=120), "
            "Item4(w=15,v=80), Item5(w=25,v=90), Item6(w=35,v=150), "
            "Item7(w=12,v=70), Item8(w=22,v=110), Item9(w=18,v=95), "
            "Item10(w=40,v=200), Item11(w=28,v=130), Item12(w=33,v=160), "
            "Item13(w=9,v=45), Item14(w=17,v=85), Item15(w=50,v=220)."
        ),
    },
    "hard": {
        "capacity": 120,
        "items": [
            (10, 60), (20, 100), (30, 120), (15, 80), (25, 90),
            (35, 150), (12, 70), (22, 110), (18, 95), (40, 200),
            (28, 130), (33, 160), (9, 45), (17, 85), (50, 220),
            (11, 55), (24, 115), (38, 175), (16, 78), (45, 210),
        ],
        "description": (
            "Capacity = 120 (tight). Items 1-20: "
            "Item1(w=10,v=60), Item2(w=20,v=100), Item3(w=30,v=120), "
            "Item4(w=15,v=80), Item5(w=25,v=90), Item6(w=35,v=150), "
            "Item7(w=12,v=70), Item8(w=22,v=110), Item9(w=18,v=95), "
            "Item10(w=40,v=200), Item11(w=28,v=130), Item12(w=33,v=160), "
            "Item13(w=9,v=45), Item14(w=17,v=85), Item15(w=50,v=220), "
            "Item16(w=11,v=55), Item17(w=24,v=115), Item18(w=38,v=175), "
            "Item19(w=16,v=78), Item20(w=45,v=210)."
        ),
    },
}

VALID_METHODS = {
    "dynamic programming", "greedy algorithm", "branch and bound",
    "backtracking", "genetic algorithm", "simulated annealing",
    "fractional knapsack", "0/1 knapsack", "meet in the middle",
    "depth first search", "breadth first search", "linear programming",
    "integer programming", "beam search", "ant colony optimization",
    "particle swarm optimization", "random search", "exhaustive search",
    "divide and conquer", "memoization", "tabulation",
}

FIELDNAMES = [
    "system", "instance", "n", "run", "calls",
    "best_value", "best_weight", "best_items", "best_method",
    "hallucination_count", "hallucinated_methods",
]

call_count = 0

# ── Checkpointing ─────────────────────────────────────────────────────────────


def load_completed_runs(csv_path: str) -> set:
    """Return set of (instance, n, run) tuples already saved to CSV."""
    completed = set()
    if not os.path.exists(csv_path):
        return completed
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                completed.add(
                    (row["instance"], int(row["n"]), int(row["run"])))
            except (KeyError, ValueError):
                pass
    return completed


def open_csv_for_append(csv_path: str):
    """Open CSV in append mode; write header only if file is new/empty."""
    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    f = open(csv_path, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    if not file_exists:
        writer.writeheader()
    return f, writer

# ── Rate-limit sentinel ───────────────────────────────────────────────────────


class RateLimitExhausted(Exception):
    pass

# ── LLM call ─────────────────────────────────────────────────────────────────


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

# ── Helpers ───────────────────────────────────────────────────────────────────


def parse_methods(text: str, n: int) -> list:
    methods = []
    for line in text.splitlines():
        clean = re.sub(r"^[\s\*\-\•\d\.\)\#]+", "", line).strip(" :.*_")
        clean = re.sub(r"\*+|_+", "", clean).strip()
        if not clean or len(clean) < 4 or len(clean) > 80:
            continue
        if re.search(r"[\[\]{},]{3,}", clean):
            continue
        methods.append(clean)
        if len(methods) == n:
            break
    return methods


def is_hallucinated(method_name: str) -> bool:
    name_lower = method_name.lower().strip()
    return not any(v in name_lower or name_lower in v for v in VALID_METHODS)


def extract_item_list(text: str) -> list:
    matches = re.findall(r"\[[\d,\s]+\]", text)
    if not matches:
        return []
    try:
        return [int(x) for x in re.findall(r"\d+", matches[-1])]
    except Exception:
        return []


def evaluate(items_chosen: list, instance: dict) -> tuple:
    items = instance["items"]
    cap = instance["capacity"]
    tw = sum(items[i-1][0] for i in items_chosen if 1 <= i <= len(items))
    tv = sum(items[i-1][1] for i in items_chosen if 1 <= i <= len(items))
    return tv, tw, tw <= cap


def phase5_pick_best(trajectories: list, instance: dict) -> dict:
    best = {"value": -1, "weight": 0, "items": [], "method": "none"}
    for traj in trajectories:
        chosen = list(dict.fromkeys(extract_item_list(traj["phase4_output"])))
        if not chosen:
            continue
        val, wt, feasible = evaluate(chosen, instance)
        if feasible and val > best["value"]:
            best = {"value": val, "weight": wt,
                    "items": chosen, "method": traj["method"]}
    return best

# ── Single SGE run ────────────────────────────────────────────────────────────


def run_sge(client, instance_name: str, instance: dict, n: int, run_id: int) -> dict:
    global call_count
    call_count = 0

    question = (
        f"Solve this 0/1 Knapsack problem. {instance['description']} "
        "Select items to MAXIMIZE total value without exceeding capacity. "
        "Do NOT write code."
    )

    print(f"\n\n{'#'*60}")
    print(f"  SGE | instance={instance_name} | N={n} | run={run_id}")
    print(f"{'#'*60}")

    raw = llm(client,
              question + "\n\nList few method names to solve this. "
              "One per line. No numbering, no bullets, no explanation.",
              "PHASE 1 — Exploration"
              )
    methods = parse_methods(raw, n)
    print(f"\n  Parsed methods: {methods}")

    hallucinated_names = [m for m in methods if is_hallucinated(m)]
    print(f"  Hallucinated  : {hallucinated_names}")

    if not methods:
        return {
            "system": "SGE", "instance": instance_name, "n": n, "run": run_id,
            "calls": call_count, "best_value": 0, "best_weight": 0,
            "best_items": "[]", "best_method": "none",
            "hallucination_count": 0, "hallucinated_methods": "",
        }

    trajectories = []
    for idx, method in enumerate(methods):
        t = idx + 1
        print(f"\n\n{'='*55}\n  TRAJECTORY {t}/{n}: {method}\n{'='*55}")

        steps = llm(client,
                    f"{question}\n\nMethod: {method}\n"
                    "List only the steps. One per line. No code, no explanation.",
                    f"PHASE 2 — Decomposition [T{t}]"
                    )
        action = llm(client,
                     f"{question}\n\nMethod: {method}\nSteps:\n{steps}\n\n"
                     "Apply these steps. Show your working. "
                     "End with a Python list of chosen item numbers.",
                     f"PHASE 3 — Action [T{t}]"
                     )
        phase4 = llm(client,
                     f"{question}\n\nCurrent solution:\n{action}\n\n"
                     "Check for errors. Can any swap increase value without exceeding capacity? "
                     "End with your final improved Python list of item numbers.",
                     f"PHASE 4 — Refinement [T{t}]"
                     )
        trajectories.append({"method": method, "phase4_output": phase4})

    best = phase5_pick_best(trajectories, instance)
    print(f"\n  PHASE 5 — Best: items={best['items']}, "
          f"value={best['value']}, weight={best['weight']}, method={best['method']}")

    return {
        "system":               "SGE",
        "instance":             instance_name,
        "n":                    n,
        "run":                  run_id,
        "calls":                call_count,
        "best_value":           best["value"],
        "best_weight":          best["weight"],
        "best_items":           str(best["items"]),
        "best_method":          best["method"],
        "hallucination_count":  len(hallucinated_names),
        "hallucinated_methods": "; ".join(hallucinated_names),
    }

# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    completed = load_completed_runs(CSV_OUT)
    total = len(N_VALUES) * len(INSTANCES) * RUNS
    remaining = total - len(completed)

    print(
        f"\n  Checkpoint: {len(completed)}/{total} runs done, {remaining} remaining.")
    if not remaining:
        print("  Nothing to do — all runs complete!")
        return

    # Create client here so swapping GROQ_API_KEY and re-running just works
    client = Groq(api_key=API_KEY)
    f, writer = open_csv_for_append(CSV_OUT)

    try:
        for n in N_VALUES:
            for inst_name, instance in INSTANCES.items():
                for run_id in range(1, RUNS + 1):
                    key = (inst_name, n, run_id)
                    if key in completed:
                        print(
                            f"  → Skip (done): {inst_name} N={n} run={run_id}")
                        continue

                    result = run_sge(client, inst_name, instance, n, run_id)
                    writer.writerow(result)
                    f.flush()
                    completed.add(key)

                    print(f"\n  ✓ [{len(completed)}/{total}] SGE | {inst_name} | "
                          f"N={n} | run={run_id} | value={result['best_value']} "
                          f"| calls={result['calls']} "
                          f"| hallucinations={result['hallucination_count']}")

    except RateLimitExhausted:
        done = len(completed)
        print(f"\n\n{'!'*60}")
        print(f"  RATE LIMIT EXHAUSTED — stopped cleanly.")
        print(f"  Progress : {done}/{total} runs saved to {CSV_OUT}")
        print(f"  Remaining: {total - done} runs")
        print(f"\n  TO RESUME:")
        print(f"  1. Open this file and replace API_KEY with a fresh key.")
        print(f"  2. Run the script again — completed runs are skipped automatically.")
        print(f"{'!'*60}")

    except KeyboardInterrupt:
        print(f"\n\n  Stopped by user. {len(completed)}/{total} runs saved.")
        print(f"  Re-run to continue from where you left off.")

    finally:
        f.close()

    print(f"\n  Results saved to {CSV_OUT}")


if __name__ == "__main__":
    main()
