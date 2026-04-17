# Enhancing SGE with Adaptive Pruning and RAG for Combinatorial Problems

This repository contains code for two extensions of the Self-Guiding Exploration (SGE) algorithm (NeurIPS 2024) applied to combinatorial optimization:

- **SGE-AP (Adaptive Pruning):** reduces LLM calls by pruning weak trajectories before the expensive refinement phase.
- **RAG-SGE:** replaces the exploration phase with a deterministic knowledge-base lookup, reducing hallucination and removing LLM dependency for method selection.

The code includes experiments for **Vehicle Routing Problem (VRP)** and **0/1 Knapsack** (fixed instances and/or random instances depending on the script).

## Repository layout

- [sge/](sge/) — core SGE implementation and adaptive pruning variant
  - [sge/sge.py](sge/sge.py) — base SGE pipeline (explore → decompose → resolve → refine → integrate)
  - [sge/sge_pruned.py](sge/sge_pruned.py) — SGE-AP (pruning checkpoint after Phase 3)
  - [sge/problems.py](sge/problems.py) — VRP instance generator + OR-Tools optimal solver (for gap computation)
  - [sge/llm.py](sge/llm.py) — Groq LLM wrapper (reads `GROQ_API_KEY`)
- [experiments/](experiments/) — scripts to reproduce baseline vs pruning results (VRP, random instances)
- [rag/](rag/) — RAG-SGE experiment scripts (VRP + Knapsack) and [rag/knowledge_base.json](rag/knowledge_base.json)

## Requirements

- Python 3.9+ recommended
- A Groq API key (environment variable `GROQ_API_KEY`) for scripts that call the LLM

Install Python dependencies (minimal set used by this repo):

```bash
pip install groq python-dotenv numpy scipy ortools
```

## API key setup

For the [experiments/](experiments/) + [sge/](sge/) code path, the LLM client reads `GROQ_API_KEY` from your environment (and [experiments/run_baseline.py](experiments/run_baseline.py) / [experiments/run_pruned.py](experiments/run_pruned.py) load a local `.env` if present).

Create a `.env` file in the repo root:

```bash
GROQ_API_KEY=your_key_here
```

Notes:

- `.env` is ignored by git (see [.gitignore](.gitignore)).
- Before pushing this project to a **public** repo, make sure you are not committing any secrets.

## Running experiments (VRP — random instances)

These scripts write CSVs under a `results/` folder at the repo root.

### 1) Baseline SGE

```bash
python experiments/run_baseline.py
```

Key settings are at the top of [experiments/run_baseline.py](experiments/run_baseline.py):

- `DIFFICULTY` (`easy` / `medium` / `hard`)
- `N_VALUES` (number of explored trajectories)
- `NUM_INSTANCES` (how many random seeds to run)

### 2) SGE-AP (Adaptive Pruning)

```bash
python experiments/run_pruned.py
```

Key settings are at the top of [experiments/run_pruned.py](experiments/run_pruned.py):

- `N` (fixed number of trajectories)
- `KEEP_RATIOS` (fraction kept after Phase 3)
- `DIFFICULTY`, `NUM_INSTANCES`

### 3) Compare baseline vs pruning

```bash
python experiments/compare.py
```

This reads the generated CSVs and prints a final comparison table.

## Running RAG-SGE experiments (fixed instances)

RAG-SGE scripts use a deterministic lookup from [rag/knowledge_base.json](rag/knowledge_base.json) for Phase 1 (method selection) and then call the LLM to execute/refine the chosen method steps.

- VRP: [rag/improved_rag_sge_vrp.py](rag/improved_rag_sge_vrp.py)
- Knapsack: [rag/improved_rag_sge_knapsack.py](rag/improved_rag_sge_knapsack.py)

Run them from the repo root, for example:

```bash
python rag/improved_rag_sge_vrp.py
python rag/improved_rag_sge_knapsack.py
```

## Important note for public GitHub repos

All scripts in this repo are intended to read your Groq key from the `GROQ_API_KEY` environment variable (optionally via a local `.env`). Do **not** commit API keys to a public repository.

If GitHub blocks a push due to secret scanning:

1. Remove the secret from the file(s).
2. Rewrite the offending commit(s) (e.g., amend/rebase) so the secret is not present in git history.
3. Rotate the key in your provider dashboard if you suspect it may have been exposed.

## Attribution

This codebase is based on the SGE idea from “Self-Guiding Exploration for Combinatorial Problems” (NeurIPS 2024) and implements research-oriented extensions (adaptive pruning and knowledge-base retrieval for exploration).
