
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import csv
import numpy as np
from collections import defaultdict
from dotenv import load_dotenv
from sge import LLMClient, SGE
from sge.problems import INSTANCES, create_instance

load_dotenv()


##### CONFIG #####
N_VALUES = [2, 4, 6]
DIFFICULTY = "easy"
NUM_INSTANCES = 10
RESULTS_FILE = os.path.join(
    os.path.dirname(__file__), '..',
    f'results/baseline_{DIFFICULTY}.csv')
FIELDNAMES = [
    'instance_id', 'difficulty', 'N', 'num_cities',
    'call_count', 'model_cost', 'optimal_cost', 'gap_pct',
    'methods_used', 'trajectory_scores'
]


def load_completed(filepath):
    completed = set()
    if not os.path.exists(filepath):
        return completed
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                completed.add((int(row['instance_id']), int(row['N'])))
            except:
                continue
    return completed


def ensure_results_dir():
    os.makedirs(os.path.join(
        os.path.dirname(__file__), '..', 'results'), exist_ok=True)


if __name__ == "__main__":
    ensure_results_dir()
    completed = load_completed(RESULTS_FILE)
    print(f"Difficulty: {DIFFICULTY}")
    print(f"Cities: {INSTANCES[DIFFICULTY]['num_cities']}")
    print(f"Already completed: {len(completed)} runs")

    write_header = not os.path.exists(RESULTS_FILE)
    llm = LLMClient(sleep_between_calls=12)

    with open(RESULTS_FILE, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

        for N in N_VALUES:
            for instance_id in range(NUM_INSTANCES):
                if (instance_id, N) in completed:
                    print(f"Skipping N={N} instance={instance_id}")
                    continue

                print(f"\n{'='*60}")
                print(
                    f"Difficulty={DIFFICULTY} | N={N} | instance={instance_id}")
                print(f"{'='*60}")

                try:
                    problem = create_instance(DIFFICULTY, seed=instance_id)
                    num_cities = INSTANCES[DIFFICULTY]['num_cities']

                    opt_route, opt_cost = problem.solve_optimal()
                    print(f"Optimal: {opt_route} cost={opt_cost}")

                    sge = SGE(llm, problem, N=N, verbose=True)
                    Q = problem.create_prompt()
                    route, cost, call_count, traj_info = sge.run(Q)

                    gap = problem.compute_gap(route)
                    methods = [t['method'] for t in traj_info]
                    scores = [t['score'] for t in traj_info]

                    print(
                        f"\nResult: route={route} cost={cost} gap={gap:.2f}%")
                    print(f"Calls: {call_count}")

                    writer.writerow({
                        'instance_id': instance_id,
                        'difficulty': DIFFICULTY,
                        'N': N,
                        'num_cities': num_cities,
                        'call_count': call_count,
                        'model_cost': cost,
                        'optimal_cost': opt_cost,
                        'gap_pct': round(gap, 2),
                        'methods_used': ' | '.join(methods),
                        'trajectory_scores': ' | '.join(
                            str(s) for s in scores)
                    })
                    f.flush()
                    completed.add((instance_id, N))

                except Exception as e:
                    print(f"FAILED N={N} instance={instance_id}: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    import time
                    time.sleep(15)

    # Summary from full CSV
    print("\n" + "="*70)
    print(f"SUMMARY — difficulty={DIFFICULTY}")
    print("="*70)
    print(f"{'N':<5} {'Count':<8} {'Calls':<10} {'Mean Gap%':<12}"
          f"{'Std Gap%':<12} {'Min Gap%':<12} {'Max Gap%':<12}")
    print("-"*70)

    all_results = defaultdict(list)
    with open(RESULTS_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                all_results[int(row['N'])].append({
                    'calls': int(row['call_count']),
                    'gap': float(row['gap_pct'])
                })
            except:
                continue

    for N_val in sorted(all_results.keys()):
        data = all_results[N_val]
        gaps = [d['gap'] for d in data if d['gap'] >= 0]
        calls = [d['calls'] for d in data]
        if not gaps:
            continue
        print(f"{N_val:<5} {len(gaps):<8} {np.mean(calls):<10.1f}"
              f"{np.mean(gaps):<12.2f} {np.std(gaps):<12.2f}"
              f"{np.min(gaps):<12.2f} {np.max(gaps):<12.2f}")
    print("="*70)
