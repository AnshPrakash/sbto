from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf

from sbto.data.constants import KEY_PD_KNOTS, KEY_PD_TARGET
from sbto.data.save import select_best_trajectory
from sbto.utils.hydra import (
    get_initial_state_solver_from_controls,
    update_cfg_from_warm_start,
)


def test_initial_controls_are_sampled_at_solver_knots(tmp_path):
    path = tmp_path / "controls.npz"
    controls = np.arange(10, dtype=float).reshape(5, 2)
    np.savez(path, **{KEY_PD_TARGET: controls})
    sim = SimpleNamespace(
        Nu=2,
        Nknots=3,
        scaling=SimpleNamespace(inverse=lambda value: value),
    )
    solver = SimpleNamespace(init_state=lambda mean: mean)

    mean = get_initial_state_solver_from_controls(sim, solver, path)

    assert np.array_equal(mean.reshape(3, 2), controls[[0, 2, 4]])


def test_exact_initial_knots_take_precedence(tmp_path):
    path = tmp_path / "controls.npz"
    np.savez(
        path,
        **{
            KEY_PD_TARGET: np.zeros((5, 2)),
            KEY_PD_KNOTS: np.ones((3, 2)),
        },
    )
    sim = SimpleNamespace(
        Nu=2,
        Nknots=3,
        scaling=SimpleNamespace(inverse=lambda value: value),
    )
    solver = SimpleNamespace(init_state=lambda mean: mean)

    mean = get_initial_state_solver_from_controls(sim, solver, path)

    assert np.array_equal(mean.reshape(3, 2), np.ones((3, 2)))


def test_warm_start_preserves_existing_reference(tmp_path):
    current_motion = tmp_path / "current.npz"
    warm_motion = tmp_path / "warm.npz"
    current_motion.touch()
    warm_motion.touch()
    warm_dir = tmp_path / "warm" / ".hydra"
    run_dir = tmp_path / "run" / ".hydra"
    warm_dir.mkdir(parents=True)
    run_dir.mkdir(parents=True)
    (warm_dir / "config.yaml").write_text(
        f"task:\n  cfg_ref:\n    motion_path: {warm_motion}\n"
    )
    (run_dir / "config.yaml").write_text("{}\n")
    cfg = OmegaConf.create({
        "warm_start": {"rundir": str(warm_dir.parent)},
        "task": {"cfg_ref": {"motion_path": str(current_motion)}},
    })

    update_cfg_from_warm_start(cfg, str(run_dir.parent))

    assert cfg.task.cfg_ref.motion_path == str(current_motion)


def test_best_trajectory_preserves_shared_knot_times():
    result = select_best_trajectory(
        {
            "x": np.arange(12).reshape(2, 3, 2),
            "step_knots": np.array([0, 5, 10]),
        },
        1,
    )

    assert np.array_equal(result["x"], np.arange(6, 12).reshape(3, 2))
    assert np.array_equal(result["step_knots"], [0, 5, 10])
