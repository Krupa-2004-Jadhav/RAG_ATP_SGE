"""
Clean implementation of Algorithm 1 from:
"Self-Guiding Exploration for Combinatorial Problems" (NeurIPS 2024)

Phases:
  1. Exploration  — generate N heuristic method trajectories
  2. Decomposition — break each trajectory into K subtasks
  3. Resolution    — execute each subtask sequentially
  4. Refinement    — get feedback and improve each subtask result
  5. Integration   — combine all trajectories into final answer
"""


class Trajectory:
    """Holds the state of one heuristic trajectory."""

    def __init__(self, n, method_name):
        self.n = n
        self.method_name = method_name
        self.steps = []
        self.thoughts = []     # one per subtask
        self.final_thought = ""
        self.score = None      # filled after Resolution phase
        self.pruned = False


class SGE:
    def __init__(self, llm, problem, N=4, verbose=True):
        """
        llm     : LLMClient instance
        problem : VRPProblem instance  
        N       : number of trajectories to explore
        verbose : print phase progress
        """
        self.llm = llm
        self.problem = problem
        self.N = N
        self.verbose = verbose
        self.trajectories = []

    def log(self, msg):
        if self.verbose:
            print(msg)

    def phase1_explore(self, Q):
        """
        Phase 1: Ask LLM to list heuristic methods.
        Returns list of method name strings.
        """
        self.log("\n[Phase 1] Exploration")
        prompt = (
            f"{Q}\n\n"
            "List heuristic methods that can solve this problem. "
            "Return ONLY method names, one per line, no explanations, "
            "no numbering, no bullets."
        )
        response = self.llm.predict(prompt)
        self.log(f"Raw exploration response:\n{response}")

        # Parse and clean method names
        methods = []
        skip_starts = [
            'here', 'the', 'to', 'a', 'an', 'these', 'this',
            'some', 'below', 'i', 'we', 'sure', 'heuristic',
            'heuristics', 'following', 'above', 'note', 'method',
            'methods', 'solution', 'there', 'based', 'using',
            'you', 'please', 'for', 'in', 'of'
        ]
        for line in response.splitlines():
            line = line.strip()
            if not line or len(line) < 3:
                continue
            if line[0] in ['{', '[', '(']:
                continue
            if line[0].isdigit() and len(line) > 1 and line[1] in ['.', ')', ':']:
                continue
            first_word = line.split()[0].lower().rstrip(':')
            if first_word in skip_starts:
                continue
            methods.append(line)

        methods = methods[:self.N]
        self.log(f"Cleaned methods: {methods}")

        # Create trajectory objects
        self.trajectories = [
            Trajectory(n, method) for n, method in enumerate(methods)
        ]
        return methods

    def phase2_decompose(self, Q, trajectory):
        """
        Phase 2: Decompose one trajectory into subtask steps.
        Returns list of step strings.
        """
        self.log(
            f"\n[Phase 2] Decomposing trajectory {trajectory.n}: {trajectory.method_name}")
        prompt = (
            f"{Q}\n\n"
            f"Heuristic method: {trajectory.method_name}\n\n"
            "List the exact steps to apply this method to solve the problem above. "
            "Return ONLY the steps, one per line, no explanations."
        )
        response = self.llm.predict(prompt)

        steps = [
            line.strip() for line in response.splitlines()
            if line.strip() and len(line.strip()) > 5
        ]
        trajectory.steps = steps
        self.log(f"Steps ({len(steps)}): {steps[:3]}...")
        return steps

    def phase3_resolve(self, Q, trajectory):
        """
        Phase 3: Execute subtasks sequentially.
        Each subtask builds on the previous thought.
        Returns final thought string.
        """
        self.log(f"\n[Phase 3] Resolving trajectory {trajectory.n}")
        previous_thought = ""

        for k, step in enumerate(trajectory.steps):
            prompt = (
                f"{Q}\n\n"
                f"Method: {trajectory.method_name}\n"
                f"Current step: {step}\n"
            )
            if previous_thought:
                prompt += f"\nPrevious progress:\n{previous_thought}\n"
            prompt += (
                "\nApply this step. Show your work briefly. "
                "At the end, show the current best route as a list like [0, 2, 3, 1, 4, 0]"
            )

            thought = self.llm.predict(prompt)
            trajectory.thoughts.append(thought)
            previous_thought = thought
            self.log(f"  Step {k}: {thought[:100]}...")

        trajectory.final_thought = previous_thought

        # Score this trajectory numerically
        route = self.problem.parse_solution(trajectory.final_thought)
        trajectory.score = self.problem.compute_cost(route)
        self.log(f"  Trajectory {trajectory.n} score: {trajectory.score}")
        return trajectory.final_thought

    def phase4_refine(self, Q, trajectory):
        """
        Phase 4: Get feedback and refine the trajectory's solution.
        Returns refined thought string.
        """
        self.log(f"\n[Phase 4] Refining trajectory {trajectory.n}")
        prompt = (
            f"{Q}\n\n"
            f"Current solution from {trajectory.method_name}:\n"
            f"{trajectory.final_thought}\n\n"
            "Identify any improvements to this route. "
            "Return the improved route as a list like [0, 2, 3, 1, 4, 0] "
            "with no explanation."
        )
        refined = self.llm.predict(prompt)
        trajectory.final_thought = refined

        # Update score after refinement
        route = self.problem.parse_solution(refined)
        new_score = self.problem.compute_cost(route)
        self.log(f"  Refined score: {new_score} (was {trajectory.score})")
        trajectory.score = new_score
        return refined

    def phase5_integrate(self, Q, survivors):
        """
        Phase 5: Integrate surviving trajectories into final answer.
        Picks best by numeric score, not by asking LLM to compare.
        """
        self.log(f"\n[Phase 5] Integration from {len(survivors)} survivors")

        # Build candidate list with scores
        candidates = []
        for traj in survivors:
            route = self.problem.parse_solution(traj.final_thought)
            cost = self.problem.compute_cost(route)
            candidates.append((cost, route, traj))
            self.log(
                f"  Trajectory {traj.n} ({traj.method_name}): cost={cost}")

        # Pick best by numeric cost (deterministic, no LLM needed)
        best_cost, best_route, best_traj = min(candidates, key=lambda x: x[0])
        self.log(
            f"  Best: trajectory {best_traj.n} ({best_traj.method_name}) cost={best_cost}")

        return best_route, best_cost

    def run(self, Q):
        """
        Run full SGE pipeline.
        Returns (best_route, best_cost, call_count, trajectory_info)
        """
        self.llm.reset_counter()

        # Phase 1
        methods = self.phase1_explore(Q)
        if not self.trajectories:
            return [], 99999, self.llm.get_call_count(), []

        # Phases 2 + 3 for all trajectories
        for traj in self.trajectories:
            self.phase2_decompose(Q, traj)
            self.phase3_resolve(Q, traj)

        survivors = self.trajectories  # no pruning in base SGE

        # Phase 4 for all survivors
        for traj in survivors:
            self.phase4_refine(Q, traj)

        # Phase 5
        best_route, best_cost = self.phase5_integrate(Q, survivors)
        call_count = self.llm.get_call_count()

        traj_info = [
            {
                'n': t.n,
                'method': t.method_name,
                'score': t.score,
                'pruned': t.pruned
            }
            for t in self.trajectories
        ]

        return best_route, best_cost, call_count, traj_info
