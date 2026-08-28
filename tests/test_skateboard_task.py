from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from sbto.tasks.cost import lower_bound_cost_nb
from sbto.utils.extract_ref import ReferenceMotion


def test_lower_bound_cost_only_penalizes_violations():
    values = np.array([[[0.5], [0.7]]])
    bound = np.full((2, 1), 0.65)
    weights = np.full((2, 1), 100.0)

    assert np.isclose(lower_bound_cost_nb(values, bound, weights)[0], 2.25)


def test_skateboard_task_uses_full_articulated_state(tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(repo)
    with initialize_config_dir(config_dir=str(repo / "sbto/conf"), version_base=None):
        cfg = compose(config_name="config", overrides=["task=g1/skateboard_ref"])

    sim = instantiate(cfg.task.sim)
    qpos = np.tile(sim.mj_scene.mj_model.qpos0, (40, 1))
    qpos[:, 2] = 0.8
    qpos[:, sim.mj_scene.obj_pos_adr] = [0.0, 0.0, 0.14]
    ref_path = tmp_path / "skateboard_reference.npz"
    np.savez(ref_path, qpos=qpos, fps=100.0)
    cfg.task.cfg_ref.motion_path = str(ref_path)
    cfg.task.cfg_ref.flip_quat_pos = False
    cfg.task.cfg.torso_quat_weight_terminal = 123.0
    cfg.task.cfg.torso_linvel_weight_terminal = 234.0
    cfg.task.cfg.board_quat_weight_terminal = 456.0
    cfg.task.cfg.board_v_weight_terminal = 567.0
    cfg.task.cfg.foot_board_position_weight_terminal = 678.0

    compute_sensor_data = ReferenceMotion.compute_sensor_data

    def inject_multi_contact_count(ref, sensor_names):
        compute_sensor_data(ref, sensor_names)
        if "left_foot_deck" in sensor_names:
            ref.sensor_data["left_foot_deck"][0, 0] = 2

    monkeypatch.setattr(
        ReferenceMotion, "compute_sensor_data", inject_multi_contact_count
    )

    task = instantiate(cfg.task, sim=sim)

    assert (sim.mj_scene.Nq, sim.mj_scene.Nv, sim.mj_scene.Nu) == (49, 47, 29)
    assert task.ref.x.shape == (40, 96)
    assert set(np.unique(task.contact_plan)) <= {0, 1}
    assert np.all(task.contact_plan[-4:] == 1)
    assert cfg.task.sim.mj_scene.cfg.xml_contact_pairs_path[0].endswith("full.xml")
    assert sim.mj_scene.mj_model.sensor("robot_floor").id >= 0
    for name in task.INVALID_BOARD_FLOOR_SENSORS:
        assert sim.mj_scene.mj_model.sensor(name).id >= 0
    assert sim.mj_scene.mj_model.sensor("left_foot_pos_board").id >= 0
    assert any(name.startswith("robot_floor_") for name in task._costs_names)
    assert any(name.startswith("deck_floor+") for name in task._costs_names)
    assert any(
        name.startswith("left_foot_pos_board+right_foot_pos_board_")
        for name in task._costs_names
    )
    assert sum(name.startswith("global_pos_torso_") for name in task._costs_names) == 2
    torso_quat_cost = next(
        i for i, name in enumerate(task._costs_names)
        if name.startswith("orientation_torso_")
    )
    assert np.all(task._cost_terms["w"][torso_quat_cost][-1] == 123.0)
    torso_linvel_cost = next(
        i
        for i, name in enumerate(task._costs_names)
        if name.startswith("global_linvel_torso_")
    )
    assert np.all(task._cost_terms["w"][torso_linvel_cost][-1] == 234.0)
    board_quat_cost = task._costs_names.index("board_quat")
    assert np.all(task._cost_terms["w"][board_quat_cost][-1] == 456.0)
    board_velocity_cost = task._costs_names.index("board_velocity")
    assert np.all(task._cost_terms["w"][board_velocity_cost][-1] == 567.0)
    foot_board_cost = next(
        i
        for i, name in enumerate(task._costs_names)
        if name.startswith("left_foot_pos_board+right_foot_pos_board_")
    )
    assert np.all(task._cost_terms["w"][foot_board_cost][-1] == 678.0)
    for joint_name in ("trj0", "whj0", "whj1", "trj1", "whj2", "whj3"):
        address = int(sim.mj_scene.mj_model.joint(joint_name).qposadr[0])
        assert np.all(task.ref.x[:, address] == 0.0)
