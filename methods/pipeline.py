"""The one composable SGE pipeline (CLAUDE.md §4). Every method in
methods/registry.py is this same five-phase pipeline with different switches:
method-source (llm explore vs. retrieval lookup) and prune (on/off). Base SGE
and KB-SGE share the exact same Resolve/Refine prompt templates - the only
difference is where the method list and steps come from.
"""

import math

import config
from methods import registry, retrieval

_SKIP_FIRST_WORDS = {
    "here", "the", "to", "a", "an", "these", "this", "some", "below", "i",
    "we", "sure", "heuristic", "heuristics", "following", "above", "note",
    "method", "methods", "solution", "there", "based", "using", "you",
    "please", "for", "in", "of",
}


class Trajectory:
    def __init__(self, idx, name, steps=None):
        self.idx = idx
        self.name = name
        self.steps = steps or []
        self.resolve_text = ""
        self.resolve_solution = None
        self.resolve_score = None
        self.refine_text = ""
        self.refine_solution = None
        self.refine_score = None
        self.pruned = False

    def final_score(self):
        return self.refine_score if self.refine_score is not None else self.resolve_score

    def final_solution(self):
        return self.resolve_solution if self.refine_score is None else self.refine_solution


def parse_method_names(text, n):
    """Heuristic line-based parse of a free-text method list. Pads with
    placeholder names if the model returns fewer than n so that the call-count
    formula (which assumes exactly n trajectories) always holds.
    """
    names = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or len(line) < 3:
            continue
        if line[0] in "{[(":
            continue
        if line[0].isdigit() and len(line) > 1 and line[1] in ".):":
            continue
        first_word = line.split()[0].lower().rstrip(":")
        if first_word in _SKIP_FIRST_WORDS:
            continue
        names.append(line)

    names = names[:n]
    for i in range(len(names), n):
        names.append(f"Heuristic Method {i + 1}")
    return names


def parse_steps(text):
    steps = [line.strip() for line in (text or "").splitlines() if len(line.strip()) > 5]
    return steps or ["Apply the method directly to produce a solution."]


def count_hallucinated(names, known_names_lower):
    return sum(1 for name in names if name.lower() not in known_names_lower)


def _explore_prompt(problem, n):
    return (
        f"{problem.create_prompt()}\n\n"
        f"List {n} heuristic methods that could solve this problem. "
        "Return ONLY method names, one per line, no explanations, no numbering, no bullets."
    )


def _decompose_prompt(problem, method_name):
    return (
        f"{problem.create_prompt()}\n\n"
        f"Heuristic method: {method_name}\n\n"
        "List the exact steps to apply this method to solve the problem above. "
        "Return ONLY the steps, one per line, no explanations."
    )


def _resolve_prompt(problem, method_name, steps):
    steps_text = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(steps))
    return (
        f"{problem.create_prompt()}\n\n"
        f"Method: {method_name}\nSteps:\n{steps_text}\n\n"
        f"Apply these steps to solve the problem. Show your work briefly. "
        f"{problem.answer_format_hint()}"
    )


def _refine_prompt(problem, method_name, current_text):
    return (
        f"{problem.create_prompt()}\n\n"
        f"Current solution from {method_name}:\n{current_text}\n\n"
        "Identify any improvements to this solution. "
        f"{problem.answer_format_hint()}"
    )


def _io_prompt(problem):
    return f"{problem.create_prompt()}\n\n{problem.answer_format_hint()}"


def run(method_name, problem, llm_client, kb, n=config.N, keep_ratio=config.KEEP_RATIO):
    """Runs one full pipeline execution. Returns a result dict with everything
    run/run_experiment.py needs for one CSV row.
    """
    cfg = registry.METHODS[method_name]
    calls_before = llm_client.count()
    prompt_tokens_before = llm_client.prompt_tokens
    completion_tokens_before = llm_client.completion_tokens

    hallucinated_count = 0
    methods_used = []
    pruned_methods = []
    trajectories_info = []

    if method_name == "io":
        text = llm_client.predict(_io_prompt(problem), temperature=config.TEMPERATURE)
        solution = problem.parse_solution(text)
    else:
        if cfg["source"] == "llm":
            explore_text = llm_client.predict(_explore_prompt(problem, n), temperature=config.TEMPERATURE)
            names = parse_method_names(explore_text, n)
            known = retrieval.known_method_names(problem.kb_key, kb)
            hallucinated_count = count_hallucinated(names, known)
            trajectories = [Trajectory(i, name) for i, name in enumerate(names)]
        else:  # retrieval
            selected = retrieval.select_methods(problem.kb_key, problem.instance, kb, n)
            trajectories = [Trajectory(i, m["name"], m["steps"]) for i, m in enumerate(selected)]

        methods_used = [t.name for t in trajectories]

        for t in trajectories:
            if cfg["source"] == "llm":
                steps_text = llm_client.predict(_decompose_prompt(problem, t.name), temperature=config.TEMPERATURE)
                t.steps = parse_steps(steps_text)
            resolve_text = llm_client.predict(
                _resolve_prompt(problem, t.name, t.steps), temperature=config.TEMPERATURE
            )
            t.resolve_text = resolve_text
            t.resolve_solution = problem.parse_solution(resolve_text)
            t.resolve_score = problem.score(t.resolve_solution)

        if cfg["prune"]:
            survivors = sorted(trajectories, key=lambda t: t.resolve_score)
            keep_n = registry.keep_count(n, keep_ratio)
            for t in survivors[keep_n:]:
                t.pruned = True
                pruned_methods.append(t.name)
            survivors = survivors[:keep_n]
        else:
            survivors = trajectories

        for t in survivors:
            refine_text = llm_client.predict(
                _refine_prompt(problem, t.name, t.resolve_text), temperature=config.TEMPERATURE
            )
            t.refine_text = refine_text
            t.refine_solution = problem.parse_solution(refine_text)
            t.refine_score = problem.score(t.refine_solution)

        # Phase 5: deterministic integration (argmin over final scores), 0 calls.
        best = min(survivors, key=lambda t: t.final_score())
        solution = best.final_solution()

        trajectories_info = [
            {
                "name": t.name,
                "resolve_score": t.resolve_score,
                "refine_score": t.refine_score,
                "pruned": t.pruned,
            }
            for t in trajectories
        ]

    calls = llm_client.count() - calls_before
    prompt_tokens = llm_client.prompt_tokens - prompt_tokens_before
    completion_tokens = llm_client.completion_tokens - completion_tokens_before
    expected = registry.expected_calls(method_name, n, keep_ratio)
    assert calls == expected, (
        f"Call count mismatch for method={method_name!r}: measured {calls}, expected {expected}"
    )

    return {
        "solution": solution,
        "feasible": problem.is_feasible(solution),
        "objective": problem.raw_objective(solution),
        "calls": calls,
        "expected_calls": expected,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "methods_used": methods_used,
        "hallucinated_count": hallucinated_count,
        "pruned_methods": pruned_methods,
        "trajectories": trajectories_info,
    }
