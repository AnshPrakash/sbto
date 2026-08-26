from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate


def test_skateboard_task_uses_full_articulated_state(tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(repo)
    with initialize_config_dir(config_dir=str(repo / "sbto/conf"), version_base=None):
        cfg = compose(config_name="config", overrides=["task=g1/skateboard_ref"])

    sim = instantiate(cfg.task.sim)
    qpos = np.tile(sim.mj_scene.mj_model.qpos0, (5, 1))
    qpos[:, 2] = 0.8
    qpos[:, sim.mj_scene.obj_pos_adr] = [0.0, 0.0, 0.14]
    ref_path = tmp_path / "skateboard_reference.npz"
    np.savez(ref_path, qpos=qpos, fps=100.0)
    cfg.task.cfg_ref.motion_path = str(ref_path)
    cfg.task.cfg_ref.flip_quat_pos = False

    task = instantiate(cfg.task, sim=sim)

    assert (sim.mj_scene.Nq, sim.mj_scene.Nv, sim.mj_scene.Nu) == (49, 47, 29)
    assert task.ref.x.shape == (5, 96)
    assert np.all(task.contact_plan[-4:] == 1)
    for joint_name in ("trj0", "whj0", "whj1", "trj1", "whj2", "whj3"):
        address = int(sim.mj_scene.mj_model.joint(joint_name).qposadr[0])
        assert np.all(task.ref.x[:, address] == 0.0)
