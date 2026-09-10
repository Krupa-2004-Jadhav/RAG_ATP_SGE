"""Shared Problem interface. Every problem plugs into the one pipeline through
these five methods only - the LLM and the trajectory-scoring function see just
create_prompt()/answer_format_hint() and parse_solution(); the optimum is never
exposed through this interface (invariant #3, no optimum leakage).
"""

from abc import ABC, abstractmethod


class Problem(ABC):
    kb_key = None  # "VRP" or "Knapsack" - key into core/knowledge_base.json

    @abstractmethod
    def create_prompt(self):
        """Full problem description (coordinates/items etc). No optimum inside."""

    @abstractmethod
    def answer_format_hint(self):
        """One line telling the model exactly how to format its final answer."""

    @abstractmethod
    def parse_solution(self, text):
        """Extract a solution (opaque to the pipeline) from raw model text.
        Must never raise - falls back to a feasible sentinel solution."""

    @abstractmethod
    def is_feasible(self, solution):
        """True if the solution satisfies the problem's hard constraints."""

    @abstractmethod
    def raw_objective(self, solution):
        """True objective in natural units (route length / packed value),
        computed even for infeasible solutions, for logging."""

    @abstractmethod
    def score(self, solution):
        """Lower-is-better internal score used for ranking/pruning/integration.
        Infeasible solutions get the sentinel penalty (config.py)."""
