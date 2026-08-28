# Joao pop-shuvit feasibility study

Date: 2026-08-28

## Outcome

No SBTO candidate passed the physical acceptance gate. The best final local
candidate is safer than the previous review candidate during its recorded
horizon, but it is **not suitable as RL training input** because it completes
only about 126 degrees of the requested 180-degree board yaw and falls during
the post-motion hold.

| Metric | Previous review candidate | Final local candidate | Gate |
|---|---:|---:|---:|
| Minimum torso height | 0.636 m | 0.637 m | >= 0.65 m |
| Maximum torso tilt | 62.5 deg | 57.9 deg | <= 45 deg |
| Board yaw error | 39.8 deg | 54.2 deg | <= 25 deg |
| Final continuous both-feet deck contact | 0.00 s | 0.76 s | >= 0.20 s |
| Foot/robot floor contact | none | none | none |
| Invalid board-floor contact | 14.4% | 0% | none |
| Stable 0.5 s hold | fail | fail | pass |

Final local run:

```text
artifacts/experiments/joao_popshuvit/
  2026_08_28__13_20_47__feasibility_first_local_seed2_cem1024/
```

The complete Hydra command and configuration are recorded inside that run's
`.hydra/` directory. Its `evaluation.yaml` is the machine-readable result.

## What the experiments established

1. The source starts mid-motion and its direct fixed-PD replay is infeasible.
   A generated nominal lead-in stance is itself stable, but it does not make
   the subsequent kinematic trick dynamically trackable.
2. The old CEM warm-start path did not guarantee evaluation of the previous
   best control sequence. It could therefore return a worse trajectory.
3. The old CEM collapse checks mixed variance and standard-deviation units,
   prematurely disabling tight local searches.
4. A single aggregate skateboard-floor sensor cannot distinguish valid wheel
   contact from deck, nose, tail, or truck strikes.
5. Large weighted penalties and more samples did not resolve the contact-mode
   discontinuity. One saved candidate matched board yaw closely while the
   humanoid had already fallen, demonstrating that the weighted scalar cost
   permits unacceptable tradeoffs.
6. Appending a reference hold does not preserve the original knot timing; the
   trick is contact-sensitive enough that the nominal warm start changes
   behavior before the new hold can stabilize it.

## Reproduce the acceptance result

From the repository root:

```bash
uv sync --extra dev

RUN=artifacts/experiments/joao_popshuvit/2026_08_28__13_20_47__feasibility_first_local_seed2_cem1024
REF=artifacts/review_only/popshuvit-v6-sbto-state-input.npz

uv run --locked python -m sbto.evaluation.skateboard "$RUN" --reference "$REF"
uv run --locked python scripts/visualize_traj.py "$RUN" --no-ref
```

The evaluator intentionally exits with status 1 because this candidate fails.

## Recommended next dependency

Do not launch a larger sweep of the same weighted single-shooting problem and
do not export the failing `best_trajectory.npz` to RL. First obtain a seed that
is already dynamically feasible in the same MuJoCo model: preferably a
successful specialist-policy rollout, or a short manually verified control
rollout covering takeoff through landing. Then use SBTO only as a tight local
refiner and require `evaluation.yaml: passed: true` before cross-model replay
and RL ingestion.

If no feasible rollout seed exists, the next optimizer change should be a
bounded work package for feasibility-first/multiple-shooting constraints, not
additional CEM seeds. The present weighted objective has now been falsified for
this source motion.
