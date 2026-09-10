"""Method registry: name -> pipeline config, and the exact call-count formulas
from CLAUDE.md §5. The runner asserts measured calls == expected calls for
every run; a mismatch is a bug, not a warning.
"""

import math

METHODS = {
    "io": {"source": None, "prune": False},
    "base_sge": {"source": "llm", "prune": False},
    "sge_ap": {"source": "llm", "prune": True},
    "kb_sge": {"source": "retrieval", "prune": False},
    "kb_sge_ap": {"source": "retrieval", "prune": True},
}


def keep_count(n, keep_ratio):
    return max(1, math.ceil(n * keep_ratio))


def expected_calls(method, n, keep_ratio):
    if method == "io":
        return 1

    cfg = METHODS[method]
    keep = keep_count(n, keep_ratio)

    if cfg["source"] == "llm" and not cfg["prune"]:
        return 1 + 3 * n
    if cfg["source"] == "llm" and cfg["prune"]:
        return 1 + 2 * n + keep
    if cfg["source"] == "retrieval" and not cfg["prune"]:
        return 2 * n
    if cfg["source"] == "retrieval" and cfg["prune"]:
        return n + keep

    raise ValueError(f"Unknown method config for {method!r}")
