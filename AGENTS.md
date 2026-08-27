# Agent Guide

## Purpose

This is a fork of Atarilab SBTO/DynaRetarget with an articulated Unitree G1
skateboard refinement task. SBTO is an offline, CPU-based MuJoCo trajectory
optimizer. It turns a kinematic reference into model-dynamically-feasible
actuator targets; it is not the reinforcement-learning training repository.
The downstream RL project lives separately at `/home/levoz/HiWi/skateboarding`.

## Setup and checks

Use Astral `uv`; do not add a parallel Conda workflow.

```bash
uv sync --extra dev
uv run pytest -q tests
uv run ruff check PATHS_TO_CHANGED_PYTHON_FILES
```

Python is selected by `.python-version` and dependencies are frozen in
`uv.lock`. SBTO samples MuJoCo rollouts on CPU, so a GPU does not accelerate
the current optimizer and is not a reason to add a GPU framework. The inherited
codebase has existing whole-repository Ruff findings; keep new or changed Python
files clean without folding a broad lint rewrite into feature work.

## Repository map

- `sbto/main.py`: Hydra entry point.
- `sbto/conf/`: solver, task, reference, and MuJoCo scene configuration.
- `sbto/tasks/g1/skateboard_ref.py`: skateboard-specific task and costs.
- `sbto/conf/task/g1/skateboard_ref.yaml`: skateboard task composition.
- `sbto/models/unitree_g1/scene_mjx_29dof_skateboard.xml`: combined model.
- `sbto/models/skateboard/skateboard_adi.xml`: articulated board body.
- `sbto/utils/extract_ref.py`: reference-state extraction and validation.
- `tests/`: reference and skateboard contract checks.
- `docs/adr/`: architectural decisions and scientific boundaries.

## Skateboard reference contract

Input is a numeric `.npz` with `qpos` shaped `(T, 49)` plus scalar `fps`.
Ordering must exactly match the combined MuJoCo model: G1 floating base and 29
joints, skateboard free joint, then six passive truck/wheel joints. MuJoCo
free-joint data is `[position, quaternion]` with `wxyz` quaternion order; use
`task.cfg_ref.flip_quat_pos=false`. Constant passive references are preserved;
moving passive references fail fast because the optimizer cannot actuate them.

## Scientific and repository guardrails

- Keep input references immutable and write outputs to a separate artifact
  directory.
- Do not commit generated trajectories, datasets, Hydra outputs, or optimizer
  checkpoints unless the user explicitly promotes a validated fixture.
- Record the input hash, commit, seed, model, solver overrides, and validation
  results for any candidate trajectory.
- A completed optimization is not enough: check finite values, board position
  and yaw tracking, landing/deck contact, foot-floor contact, and torso tilt.
- Treat candidates as review-only until cross-model replay and downstream
  project gates pass. Never describe model feasibility as hardware safety.
- Preserve unrelated worktree changes and keep changes scoped to the active
  task. Add focused tests for state-layout or contact/cost contract changes.

Read [ADR 0001](docs/adr/0001-articulated-skateboard-refinement.md) before
changing the skateboard state layout or dynamics assumptions.
