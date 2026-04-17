"""
SGE with Adaptive Pruning (SGE-AP)

Extends base SGE by adding a pruning checkpoint after Phase 3.
After all trajectories complete subtask resolution, we score them
numerically and drop the weakest ones before Phase 4 (refinement).

This directly addresses Limitation 2 from the original paper:
"it requires 87.89% more function calls than the Decomposition method"
"""

import math
from sge.sge import SGE


class SGEPruned(SGE):
    def __init__(self, llm, problem, N=4, keep_ratio=0.5, verbose=True):
        """
        keep_ratio: fraction of trajectories to keep after pruning
                    0.5 means keep top 50% (e.g. 2 of 4)
        """
        super().__init__(llm, problem, N, verbose)
        self.keep_ratio = keep_ratio

    def prune(self):
        """
        Pruning checkpoint: score all trajectories after Phase 3,
        keep top ceil(N * keep_ratio), mark rest as pruned.
        Returns list of surviving trajectories.
        """
        # Sort by score ascending (lower cost = better)
        valid = [t for t in self.trajectories if t.score is not None]
        sorted_trajs = sorted(valid, key=lambda t: t.score)

        keep_n = max(1, math.ceil(len(sorted_trajs) * self.keep_ratio))
        survivors = sorted_trajs[:keep_n]
        pruned = sorted_trajs[keep_n:]

        for t in pruned:
            t.pruned = True

        self.log(
            f"\n[PRUNING] Scores: {[(t.method_name, t.score) for t in sorted_trajs]}")
        self.log(
            f"[PRUNING] Keeping {keep_n}/{len(sorted_trajs)}: {[t.method_name for t in survivors]}")
        self.log(
            f"[PRUNING] Dropping {len(pruned)}: {[t.method_name for t in pruned]}")

        return survivors

    def run(self, Q):
        """
        Run SGE with adaptive pruning.
        Same as base SGE but prunes between Phase 3 and Phase 4.
        """
        self.llm.reset_counter()

        # Phase 1
        methods = self.phase1_explore(Q)
        if not self.trajectories:
            return [], 99999, self.llm.get_call_count(), []

        # Phases 2 + 3 for ALL trajectories
        for traj in self.trajectories:
            self.phase2_decompose(Q, traj)
            self.phase3_resolve(Q, traj)

        # ← PRUNING CHECKPOINT HERE
        survivors = self.prune()

        # Phase 4 for SURVIVORS ONLY (pruned trajectories skip this)
        for traj in survivors:
            self.phase4_refine(Q, traj)

        # Phase 5
        best_route, best_cost = self.phase5_integrate(Q, survivors)
        call_count = self.llm.get_call_count()

        traj_info = [
            {
                'n': t.n,
                'method': t.method_name,
                'score_after_phase3': t.score,
                'pruned': t.pruned
            }
            for t in self.trajectories
        ]

        return best_route, best_cost, call_count, traj_info
