# ADR 0001: Refine against an articulated skateboard model

- Status: Accepted
- Date: 2026-08-27

## Context

The original SBTO tasks refine G1 motion with either no object or a simple
free object. Skateboard motion also depends on passive truck and wheel joints,
foot-to-deck contact, board-to-floor contact, and a controlled landing. Treating
the board as a rigid box would optimize against the wrong dynamics.

## Decision

Use the bundled articulated G1-and-skateboard MuJoCo model for both reference
extraction and rollout. A skateboard reference is a numeric `.npz` containing
model-ordered `qpos` with shape `(T, 49)` and scalar `fps`:

1. G1 floating base and 29 actuated joints.
2. Skateboard free joint.
3. Six passive truck and wheel joints.

The task tracks robot and board motion, encourages foot-to-deck contact, and
penalizes foot-to-floor contact. Passive joint reference values are preserved
when constant. A moving passive-joint reference is rejected because SBTO only
optimizes the 29 robot actuator targets and cannot directly command those six
joints.

## Consequences

- Feasibility is relative to this MuJoCo model, contact setup, timestep, and
  solver configuration; it is not a hardware-safety claim.
- An optimizer result must pass finite-value, board tracking, orientation,
  landing/contact, and torso-stability checks before downstream use.
- Cross-model replay is required before promoting a trajectory to the RL
  training pipeline.
- Generated datasets, optimizer runs, and candidate `.npz` files remain
  uncommitted unless explicitly approved as validated fixtures or releases.
- Supporting moving passive-joint targets would require a new control or
  constraint design and is intentionally out of scope.
