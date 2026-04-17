import numpy as np
import csv
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


DIFFICULTY = "easy"
BASELINE_FILE = os.path.join(
    os.path.dirname(__file__), '..',
    f'results/baseline_{DIFFICULTY}.csv')
PRUNED_FILE = os.path.join(
    os.path.dirname(__file__), '..',
    f'results/pruned_{DIFFICULTY}.csv')


def read_csv(filepath):
    rows = []
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return rows
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


if __name__ == "__main__":
    baseline_rows = read_csv(BASELINE_FILE)
    pruned_rows = read_csv(PRUNED_FILE)

    print("\n" + "="*90)
    print(f"FINAL COMPARISON TABLE — difficulty={DIFFICULTY}")
    print("="*90)
    print(f"{'Method':<32} {'N':<5} {'Count':<7} {'Calls':<10}"
          f"{'Mean Gap%':<12} {'Std Gap%':<10}"
          f"{'Call Reduction%':<18} {'Gap Change%':<12}")
    print("-"*90)

    # Baseline section
    baseline_by_N = {}
    for row in baseline_rows:
        try:
            N = int(row['N'])
            if N not in baseline_by_N:
                baseline_by_N[N] = []
            baseline_by_N[N].append({
                'calls': int(row['call_count']),
                'gap': float(row['gap_pct'])
            })
        except:
            continue

    for N in sorted(baseline_by_N.keys()):
        data = baseline_by_N[N]
        gaps = [d['gap'] for d in data if d['gap'] >= 0]
        calls = [d['calls'] for d in data]
        if not gaps:
            continue
        print(f"{'SGE N='+str(N):<32} {N:<5} {len(gaps):<7}"
              f"{np.mean(calls):<10.1f} {np.mean(gaps):<12.2f}"
              f"{np.std(gaps):<10.2f} {'-':<18} {'-':<12}")

    print()

    # Pruned section — compare against N=4 baseline
    n4_data = baseline_by_N.get(4, [])
    base_calls = np.mean([d['calls'] for d in n4_data]) if n4_data else 0
    base_gap = np.mean(
        [d['gap'] for d in n4_data if d['gap'] >= 0]) if n4_data else 0

    pruned_by_ratio = {}
    for row in pruned_rows:
        try:
            ratio = float(row['keep_ratio'])
            if ratio not in pruned_by_ratio:
                pruned_by_ratio[ratio] = []
            pruned_by_ratio[ratio].append({
                'calls': int(row['call_count']),
                'gap': float(row['gap_pct'])
            })
        except:
            continue

    for ratio in sorted(pruned_by_ratio.keys()):
        data = pruned_by_ratio[ratio]
        gaps = [d['gap'] for d in data if d['gap'] >= 0]
        calls = [d['calls'] for d in data]
        if not gaps:
            continue
        mean_calls = np.mean(calls)
        mean_gap = np.mean(gaps)
        call_reduction = ((base_calls - mean_calls) / base_calls * 100
                          if base_calls > 0 else 0)
        gap_change = mean_gap - base_gap

        label = f"SGE-AP keep={int(ratio*100)}% N=4"
        sign = "+" if gap_change > 0 else ""
        print(f"{label:<32} {'4':<5} {len(gaps):<7}"
              f"{mean_calls:<10.1f} {mean_gap:<12.2f}"
              f"{np.std(gaps):<10.2f} {call_reduction:<18.1f}"
              f"{sign}{gap_change:.2f}")

    print("="*90)
    print()
    print("Gap Change: positive = pruning hurt quality, "
          "negative = pruning improved quality")
    print("Call Reduction: % fewer calls vs SGE N=4 baseline")
    print()

    # Quick verdict
    if pruned_by_ratio:
        best_ratio = min(
            pruned_by_ratio.keys(),
            key=lambda r: np.mean(
                [d['gap'] for d in pruned_by_ratio[r] if d['gap'] >= 0]
            ) if pruned_by_ratio[r] else 999
        )
        best_data = pruned_by_ratio[best_ratio]
        best_calls = np.mean([d['calls'] for d in best_data])
        best_gap = np.mean(
            [d['gap'] for d in best_data if d['gap'] >= 0])
        reduction = ((base_calls - best_calls) / base_calls * 100
                     if base_calls > 0 else 0)

        print(f"Best pruning config: keep_ratio={best_ratio}")
        print(f"  Calls: {best_calls:.1f} vs {base_calls:.1f} baseline "
              f"({reduction:.1f}% reduction)")
        print(f"  Gap:   {best_gap:.2f}% vs {base_gap:.2f}% baseline "
              f"({best_gap - base_gap:+.2f}% change)")
