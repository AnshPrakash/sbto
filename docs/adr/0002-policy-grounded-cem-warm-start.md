# ADR 0002: Ground CEM with stable policy rollouts

- Status: Accepted
- Date: 2026-08-28

## Context

A motion-editor reference can describe the intended trick while beginning in
a state or control region that is difficult to reach by sampling a large,
dense CEM distribution. A policy trained on that reference provides feasible
rollouts, but its failed terminal states are evidence about the policy's
failure boundary rather than desirable trajectory targets.

## Decision

The skateboarding repository owns policy inference and exports one portable
numeric NPZ. SBTO consumes that archive without importing MJLab or policy code.
The interface contains model-ordered `qpos` and `qvel`, one selected
`dof_pd_target`, an ensemble named `dof_pd_target_candidates`, and
`action_joint_names`.

SBTO uses the first selected rollout state as the simulator initial state, the
selected stable successful rollout as the CEM mean, and the successful
ensemble's per-knot standard deviation as a diagonal covariance. Configurable
floors and ceilings prevent frozen dimensions and uncontrolled exploration.
The original motion-editor archive remains `task.cfg_ref.motion_path` and
therefore remains the tracking objective. Joint order and state dimensions
must match the SBTO MuJoCo model exactly.

Pre-failure windows remain a separate diagnostic artifact. They may motivate
cost or initialization changes, but are not silently mixed into the successful
warm-start distribution.

## Consequences

- The two repositories share a small artifact contract, not runtime
  dependencies.
- Diagonal CEM avoids a dense covariance eigendecomposition in the high-
  dimensional knot space and does not infer unsupported cross-time
  correlations from a small rollout ensemble.
- Policy success is not proof of SBTO-model feasibility; candidates still need
  SBTO acceptance and cross-model replay.
- Changing action ordering, action delay, state layout, or failure semantics
  invalidates the export contract and requires a new seed artifact.
