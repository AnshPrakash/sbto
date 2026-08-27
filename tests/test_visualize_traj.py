import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from sbto.data.constants import KEY_FULL_STATE, KEY_TIME

spec = importlib.util.spec_from_file_location(
    "visualize_traj", Path(__file__).parents[1] / "scripts" / "visualize_traj.py"
)
visualize_traj = importlib.util.module_from_spec(spec)
spec.loader.exec_module(visualize_traj)


def test_no_ref_does_not_load_reference(monkeypatch):
    model = object()
    sim = SimpleNamespace(mj_scene=SimpleNamespace(mj_model=model))
    data = {
        KEY_TIME: np.array([0.0]),
        KEY_FULL_STATE: np.zeros((1, 1)),
    }

    monkeypatch.setattr(
        visualize_traj,
        "get_config_from_rundir",
        lambda _: SimpleNamespace(task=SimpleNamespace(sim=object())),
    )
    monkeypatch.setattr(
        visualize_traj, "load_best_trajectory_from_rundir", lambda _: data
    )
    monkeypatch.setattr(
        visualize_traj,
        "instantiate_from_cfg",
        lambda _: (_ for _ in ()).throw(AssertionError("loaded reference")),
    )
    monkeypatch.setattr(visualize_traj, "instantiate", lambda _: sim, raising=False)
    monkeypatch.setattr(visualize_traj.mujoco, "MjData", lambda _: object())
    monkeypatch.setattr(visualize_traj, "visualize_trajectory", lambda *args: None)

    visualize_traj.main("run", with_ref=False)


def test_reference_path_can_be_overridden(monkeypatch):
    model = object()
    sim = SimpleNamespace(mj_scene=SimpleNamespace(mj_model=model))
    task = SimpleNamespace(ref=SimpleNamespace(time=np.array([0.0]), x=np.zeros((1, 1))))
    cfg = SimpleNamespace(
        task=SimpleNamespace(
            cfg_ref=SimpleNamespace(motion_path="old.npz"), sim=object()
        )
    )
    data = {KEY_FULL_STATE: np.zeros((1, 1))}

    monkeypatch.setattr(visualize_traj, "get_config_from_rundir", lambda _: cfg)
    monkeypatch.setattr(
        visualize_traj, "load_best_trajectory_from_rundir", lambda _: data
    )

    def instantiate_from_cfg(updated_cfg):
        assert updated_cfg.task.cfg_ref.motion_path == "local.npz"
        return sim, task, object(), object()

    monkeypatch.setattr(visualize_traj, "instantiate_from_cfg", instantiate_from_cfg)
    monkeypatch.setattr(visualize_traj.mujoco, "MjData", lambda _: object())
    monkeypatch.setattr(
        visualize_traj, "visualize_trajectory_with_reference", lambda *args: None
    )

    visualize_traj.main("run", reference_path="local.npz")
