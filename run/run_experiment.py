"""Unified runner (CLAUDE.md §9). Every method goes through this one script and
writes to one CSV with `method` and `size` columns - no per-method bespoke
scripts. Checkpointed/resumable: safe to Ctrl-C and re-run.
"""

import argparse
import csv
import json
import os

import config
from core import instances as instances_mod
from methods import pipeline, registry
from problems.knapsack import KnapsackProblem
from problems.vrp import VRPProblem

FIELDNAMES = [
    "problem", "size", "method", "instance_id", "repeat", "N", "keep_ratio",
    "calls", "expected_calls", "prompt_tokens", "completion_tokens",
    "gap_pct", "objective", "optimal", "feasible",
    "methods_used", "hallucinated_count", "pruned_methods",
]

PROBLEM_CLASSES = {"vrp": VRPProblem, "knapsack": KnapsackProblem}


def build_problem(problem_name, instance):
    return PROBLEM_CLASSES[problem_name](instance)


def optimal_value(problem_name, instance):
    if problem_name == "vrp":
        return instance["optimal_cost"]
    return instance["optimal_value"]


def compute_gap_pct(problem_name, objective, optimal):
    if optimal is None:
        return None
    if problem_name == "vrp":
        if optimal <= 0:
            return 0.0
        return max(0.0, (objective - optimal) / optimal * 100)
    # knapsack: maximize value, so gap is how far below optimal we landed
    if optimal <= 0:
        return 0.0
    return max(0.0, (optimal - objective) / optimal * 100)


def load_completed_keys(csv_path):
    completed = set()
    if not os.path.exists(csv_path):
        return completed
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                key = (
                    row["problem"], int(row["size"]), row["method"],
                    int(row["instance_id"]), int(row["repeat"]),
                )
                completed.add(key)
            except (KeyError, ValueError):
                continue
    return completed


def open_csv_for_append(csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    f = open(csv_path, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    if not file_exists:
        writer.writeheader()
    return f, writer


def run_matrix(problem_name, sizes, methods, repeats, llm_client, kb, csv_path, dry_run=False):
    completed = load_completed_keys(csv_path)
    f, writer = open_csv_for_append(csv_path)

    try:
        for size in sizes:
            instance_list = instances_mod.load_or_generate(problem_name, size)
            for method in methods:
                for instance in instance_list:
                    instance_id = instance["instance_id"]
                    for repeat in range(1, repeats + 1):
                        key = (problem_name, size, method, instance_id, repeat)
                        if key in completed:
                            continue

                        problem = build_problem(problem_name, instance)
                        llm_client.reset()
                        result = pipeline.run(method, problem, llm_client, kb)

                        optimal = optimal_value(problem_name, instance)
                        gap_pct = compute_gap_pct(problem_name, result["objective"], optimal)

                        row = {
                            "problem": problem_name,
                            "size": size,
                            "method": method,
                            "instance_id": instance_id,
                            "repeat": repeat,
                            "N": config.N,
                            "keep_ratio": config.KEEP_RATIO,
                            "calls": result["calls"],
                            "expected_calls": result["expected_calls"],
                            "prompt_tokens": result["prompt_tokens"],
                            "completion_tokens": result["completion_tokens"],
                            "gap_pct": gap_pct,
                            "objective": result["objective"],
                            "optimal": optimal,
                            "feasible": result["feasible"],
                            "methods_used": json.dumps(result["methods_used"]),
                            "hallucinated_count": result["hallucinated_count"],
                            "pruned_methods": json.dumps(result["pruned_methods"]),
                        }
                        writer.writerow(row)
                        f.flush()
                        completed.add(key)

                        print(
                            f"[{problem_name}/size={size}/{method}] instance={instance_id} "
                            f"repeat={repeat} calls={result['calls']} gap={gap_pct} "
                            f"feasible={result['feasible']}"
                        )
    finally:
        f.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", required=True, choices=["vrp", "knapsack"])
    parser.add_argument("--size", required=True, help="instance size, or 'all' to sweep config.SIZES")
    parser.add_argument("--method", required=True, help="method name, or 'all' for every method in config.METHODS")
    parser.add_argument("--repeats", type=int, default=config.REPEATS)
    parser.add_argument("--dry-run", action="store_true", help="use the mock LLM client - zero tokens")
    return parser.parse_args()


def main():
    args = parse_args()

    sizes = config.SIZES[args.problem] if args.size == "all" else [int(args.size)]
    methods = config.METHODS if args.method == "all" else [args.method]

    with open(config.KNOWLEDGE_BASE_PATH, "r") as kb_file:
        kb = json.load(kb_file)

    if args.dry_run:
        from core.llm_client import MockLLMClient
        llm_client = MockLLMClient()
        csv_path = config.RESULTS_CSV_PATH.replace(".csv", "_dryrun.csv")
    else:
        from core.llm_client import LLMClient
        llm_client = LLMClient()
        csv_path = config.RESULTS_CSV_PATH

    run_matrix(args.problem, sizes, methods, args.repeats, llm_client, kb, csv_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
