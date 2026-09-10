"""Single source of truth for all constants. See CLAUDE.md §12."""

MODEL = "llama-3.3-70b-versatile"
TEMPERATURE = 0.7
SLEEP_BETWEEN_CALLS = 6

N = 4                      # trajectories
KEEP_RATIO = 0.5
REPEATS = 5
NUM_INSTANCES = 10

SIZES = {"vrp": [5, 8, 10], "knapsack": [10, 15, 20]}   # swept, smallest-first

HEADLINE_SIZE = {"vrp": None, "knapsack": None}          # set from Milestone 2 data

SEED_BASE = 1234           # instance generation seeds = SEED_BASE + instance_id

METHODS = ["io", "base_sge", "sge_ap", "kb_sge", "kb_sge_ap"]

# Sentinel penalties for infeasible solutions (methods/pipeline.py scoring).
VRP_INFEASIBLE_COST = 10 ** 9
KNAPSACK_INFEASIBLE_SCORE = 0

RESULTS_DIR = "results"
INSTANCES_PATH_TEMPLATE = "results/instances_{problem}_{size}.json"
RESULTS_CSV_PATH = "results/results.csv"
KNOWLEDGE_BASE_PATH = "core/knowledge_base.json"
