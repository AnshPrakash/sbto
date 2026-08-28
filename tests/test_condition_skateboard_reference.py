import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location(
    "condition_skateboard_reference",
    Path(__file__).parents[1] / "scripts" / "condition_skateboard_reference.py",
)
conditioner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conditioner)


def test_conditioned_reference_has_continuous_complete_state():
    sim = conditioner.load_skateboard_sim()
    model = sim.mj_scene.mj_model
    source = np.tile(model.qpos0, (3, 1))
    qpos, qvel, static_frames = conditioner.build_conditioned_reference(
        sim,
        source,
        np.zeros((3, model.nv)),
        fps=50.0,
        static_duration=0.04,
        blend_duration=0.04,
        terminal_hold_duration=0.04,
        root_height_above_board=0.784,
    )

    assert qpos.shape == (8, model.nq)
    assert qvel.shape == (8, model.nv)
    assert static_frames == 2
    assert np.isfinite(qpos).all()
    assert np.isfinite(qvel).all()
    free_joints = np.flatnonzero(model.jnt_type == conditioner.mujoco.mjtJoint.mjJNT_FREE)
    for joint_id in free_joints:
        address = model.jnt_qposadr[joint_id] + 3
        assert np.allclose(np.linalg.norm(qpos[:, address : address + 4], axis=1), 1.0)


def test_conditioned_controls_pad_prior_dynamic_solution():
    sim = conditioner.load_skateboard_sim()
    stance = sim.mj_scene.mj_model.qpos0.copy()
    source = np.ones((3, sim.Nu))
    controls = conditioner.build_conditioned_controls(
        sim, stance, source, 0.02, 0.02, 0.02
    )

    assert controls.shape == (9, sim.Nu)
    assert np.allclose(controls[-2:], 1.0)


def test_zero_leadin_preserves_original_initial_state():
    sim = conditioner.load_skateboard_sim()
    source = np.tile(sim.mj_scene.mj_model.qpos0, (3, 1))
    source[0, 0] = 1.0
    qpos, _, static_frames = conditioner.build_conditioned_reference(
        sim,
        source,
        np.zeros((3, sim.mj_scene.mj_model.nv)),
        50.0,
        0.0,
        0.0,
        0.04,
        0.784,
    )

    assert static_frames == 0
    assert np.array_equal(qpos[:3], source)
