from sge.problems import INSTANCES, create_instance
from sge import LLMClient, SGEPruned
from dotenv import load_dotenv
from collections import defaultdict
import numpy as np
import csv
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()


##### CONFIG #####
N = 4
KEEP_RATIOS = [0.25, 0.5, 0.75]
DIFFICULTY = "easy"
NUM_INSTANCES = 10
RESULTS_FILE = os.path.join(
    os.path.dirname(__file__), '..',
    f'results/pruned_{DIFFICULTY}.csv')
FIELDNAMES = [
    'instance_id', 'difficulty', 'N', 'keep_ratio', 'num_cities',
    'call_count', 'model_cost', 'optimal_cost', 'gap_pct',
    'survivors', 'pruned_methods', 'trajectory_scores'
]


def load_completed(filepath):
    completed = set()
    if not os.path.exists(filepath):
        return completed
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                completed.add((
                    int(row['instance_id']),
                    float(row['keep_ratio'])
                ))
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
    print(f"Already completed: {len(completed)} runs")

    write_header = not os.path.exists(RESULTS_FILE)
    llm = LLMClient(sleep_between_calls=12)

    with open(RESULTS_FILE, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

        for keep_ratio in KEEP_RATIOS:
            for instance_id in range(NUM_INSTANCES):
                if (instance_id, keep_ratio) in completed:
                    print(
                        f"Skipping ratio={keep_ratio} instance={instance_id}")
                    continue

                print(f"\n{'='*60}")
                print(f"Difficulty={DIFFICULTY} | N={N} | "
                      f"keep_ratio={keep_ratio} | instance={instance_id}")
                print(f"{'='*60}")

                try:
                    problem = create_instance(DIFFICULTY, seed=instance_id)
                    num_cities = INSTANCES[DIFFICULTY]['num_cities']

                    opt_route, opt_cost = problem.solve_optimal()
                    print(f"Optimal: cost={opt_cost}")

                    sge = SGEPruned(
                        llm, problem,
                        N=N,
                        keep_ratio=keep_ratio,
                        verbose=True
                    )
                    Q = problem.create_prompt()
                    route, cost, call_count, traj_info = sge.run(Q)

                    gap = problem.compute_gap(route)
                    survivors = [t['method'] for t in traj_info
                                 if not t['pruned']]
                    pruned = [t['method'] for t in traj_info
                              if t['pruned']]
                    scores = [t['score_after_phase3'] for t in traj_info]

                    print(f"\nResult: cost={cost} gap={gap:.2f}%")
                    print(f"Calls: {call_count}")
                    print(f"Survivors: {survivors}")
                    print(f"Pruned: {pruned}")

                    writer.writerow({
                        'instance_id': instance_id,
                        'difficulty': DIFFICULTY,
                        'N': N,
                        'keep_ratio': keep_ratio,
                        'num_cities': num_cities,
                        'call_count': call_count,
                        'model_cost': cost,
                        'optimal_cost': opt_cost,
                        'gap_pct': round(gap, 2),
                        'survivors': ' | '.join(survivors),
                        'pruned_methods': ' | '.join(pruned),
                        'trajectory_scores': ' | '.join(
                            str(s) for s in scores)
                    })
                    f.flush()
                    completed.add((instance_id, keep_ratio))

                except Exception as e:
                    print(f"FAILED ratio={keep_ratio} "
                          f"instance={instance_id}: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    import time
                    time.sleep(15)

    # Summary
    print("\n" + "="*75)
    print(f"PRUNING SUMMARY — difficulty={DIFFICULTY} N={N}")
    print("="*75)
    print(f"{'Ratio':<8} {'Count':<8} {'Calls':<10} {'Mean Gap%':<12}"
          f"{'Std Gap%':<12} {'Call Reduction%':<18}")
    print("-"*75)

    all_results = defaultdict(list)
    with open(RESULTS_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                all_results[float(row['keep_ratio'])].append({
                    'calls': int(row['call_count']),
                    'gap': float(row['gap_pct'])
                })
            except:
                continue

    # baseline calls for N=4 no pruning = keep_ratio 1.0
    # approximate: 1 explore + N*(decomp+resolve+refine) + 1 integrate
    # with K~5 steps: 1 + 4*(1+5+1) + 1 = 30 calls
    # but actual varies so we use keep_ratio=0.75 as closest to full
    all_calls = []
    for data in all_results.values():
        all_calls.extend([d['calls'] for d in data])
    max_calls = max(all_calls) if all_calls else 1

    for ratio in sorted(all_results.keys(), reverse=True):
        data = all_results[ratio]
        gaps = [d['gap'] for d in data if d['gap'] >= 0]
        calls = [d['calls'] for d in data]
        if not gaps:
            continue
        mean_calls = np.mean(calls)
        reduction = (max_calls - mean_calls) / max_calls * 100
        print(f"{ratio:<8} {len(gaps):<8} {mean_calls:<10.1f}"
              f"{np.mean(gaps):<12.2f} {np.std(gaps):<12.2f}"
              f"{reduction:<18.1f}")
    print("="*75)
