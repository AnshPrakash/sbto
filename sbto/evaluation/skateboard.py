"""Acceptance metrics for articulated-skateboard optimization rollouts."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import mujoco
import numpy as np
import yaml
from mujoco import rollout

from sbto.data.constants import KEY_FULL_STATE, KEY_OBS, KEY_PD_TARGET
from sbto.data.utils import get_config_from_rundir, load_best_trajectory_from_rundir
from sbto.main import instantiate_from_cfg


def quaternion_angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the shortest rotation angle between wxyz quaternions."""
    dot = np.abs(np.sum(a * b, axis=-1))
    return np.degrees(2.0 * np.arccos(np.clip(dot, 0.0, 1.0)))


def quaternion_yaw_deg(q: np.ndarray) -> np.ndarray:
    """Return continuous world-z yaw for a sequence of wxyz quaternions."""
    w, x, y, z = np.moveaxis(q, -1, 0)
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.degrees(np.unwrap(yaw))


def torso_tilt_deg(q: np.ndarray) -> np.ndarray:
    """Return torso up-axis tilt from world vertical for wxyz quaternions."""
    _, x, y, _ = np.moveaxis(q, -1, 0)
    up_z = 1.0 - 2.0 * (x * x + y * y)
    return np.degrees(np.arccos(np.clip(up_z, -1.0, 1.0)))


def trailing_true_duration(mask: np.ndarray, dt: float) -> float:
    """Return how long a boolean condition holds continuously at the end."""
    false_indices = np.flatnonzero(~mask)
    start = false_indices[-1] + 1 if false_indices.size else 0
    return float((mask.size - start) * dt)


def replay_observations(model: mujoco.MjModel, states: np.ndarray) -> np.ndarray:
    data = mujoco.MjData(model)
    observations = []
    for state in states:
        data.qpos[:] = state[: model.nq]
        data.qvel[:] = state[model.nq : model.nq + model.nv]
        mujoco.mj_forward(model, data)
        observations.append(data.sensordata.copy())
    return np.asarray(observations)


def invalid_board_floor_contacts(
    model: mujoco.MjModel, states: np.ndarray
) -> np.ndarray:
    """Detect deck, nose, tail, or truck strikes; wheel contact is allowed."""
    floor = model.geom("floor").id
    invalid = {
        model.geom(name).id
        for name in (
            "deck_collision",
            "nose_collision",
            "tail_collision",
            "front_truck_collision",
            "rear_truck_collision",
        )
    }
    data = mujoco.MjData(model)
    result = np.zeros(len(states), dtype=bool)
    for index, state in enumerate(states):
        data.qpos[:] = state[: model.nq]
        data.qvel[:] = state[model.nq : model.nq + model.nv]
        mujoco.mj_forward(model, data)
        result[index] = any(
            (contact.geom[0] == floor and contact.geom[1] in invalid)
            or (contact.geom[1] == floor and contact.geom[0] in invalid)
            for contact in data.contact
        )
    return result


def hold_rollout(
    model: mujoco.MjModel,
    final_state: np.ndarray,
    final_control: np.ndarray,
    duration: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    steps = round(duration / model.opt.timestep)
    controls = np.repeat(final_control[None, None], steps, axis=1)
    states, observations = rollout.rollout(
        model,
        mujoco.MjData(model),
        np.concatenate(([0.0], final_state))[None],
        controls,
    )
    return states[0, :, 1:], observations[0]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate_run(rundir: Path, reference: Path | None = None) -> dict:
    cfg = get_config_from_rundir(str(rundir))
    if cfg is None:
        raise FileNotFoundError(f"No Hydra configuration found below {rundir}")
    if reference is not None:
        cfg.task.cfg_ref.motion_path = str(reference.resolve())

    sim, task, _, _ = instantiate_from_cfg(cfg)
    model = sim.mj_scene.mj_model
    data = load_best_trajectory_from_rundir(str(rundir))
    states = np.asarray(data[KEY_FULL_STATE])
    if states.shape != task.ref.x.shape:
        raise ValueError(
            f"Rollout/reference state mismatch: {states.shape} != {task.ref.x.shape}"
        )
    observations = np.asarray(data[KEY_OBS]) if KEY_OBS in data else None
    if observations is None or observations.shape[1] != model.nsensordata:
        observations = replay_observations(model, states[1:])
    if KEY_PD_TARGET not in data:
        raise ValueError("Trajectory is missing dof_pd_target required for hold replay")
    controls = np.asarray(data[KEY_PD_TARGET])

    torso_pos = task.get_sensors_adr("global_pos_torso")
    torso_quat = task.get_sensors_adr("orientation_torso")
    deck = task.get_sensors_adr(task.DECK_CONTACT_SENSORS)
    foot_floor = task.get_sensors_adr(task.FOOT_FLOOR_SENSORS)
    board_floor = task.get_sensors_adr(task.BOARD_FLOOR_SENSOR)
    robot_floor = task.get_sensors_adr(task.ROBOT_FLOOR_SENSOR)
    landing_steps = min(
        observations.shape[0], round(cfg.task.cfg.landing_duration / model.opt.timestep)
    )

    ref_obs = replay_observations(model, task.ref.x[1:])
    board_quat = states[:, sim.mj_scene.obj_quat_adr]
    ref_board_quat = task.ref.x[:, sim.mj_scene.obj_quat_adr]
    board_yaw = quaternion_yaw_deg(board_quat)
    ref_board_yaw = quaternion_yaw_deg(ref_board_quat)
    board_pos_error = np.linalg.norm(
        states[:, sim.mj_scene.obj_pos_adr]
        - task.ref.x[:, sim.mj_scene.obj_pos_adr],
        axis=-1,
    )
    torso_error = quaternion_angle_deg(
        observations[:, torso_quat], ref_obs[:, torso_quat]
    )
    both_deck = np.all(observations[:, deck] > 0, axis=1)
    invalid_board_floor = invalid_board_floor_contacts(model, states[1:])
    hold_states, hold_obs = hold_rollout(
        model, states[-1], controls[-1]
    )
    hold_both_deck = np.all(hold_obs[:, deck] > 0, axis=1)
    hold_invalid_board_floor = invalid_board_floor_contacts(model, hold_states)
    landing = slice(-landing_steps, None)
    dt = float(model.opt.timestep)

    metrics = {
        "finite": bool(
            np.isfinite(states).all()
            and np.isfinite(observations).all()
            and np.isfinite(controls).all()
            and np.isfinite(hold_states).all()
            and np.isfinite(hold_obs).all()
        ),
        "torso_height_min_m": float(observations[:, torso_pos[2]].min()),
        "torso_height_final_m": float(observations[-1, torso_pos[2]]),
        "torso_tilt_max_deg": float(
            torso_tilt_deg(observations[:, torso_quat]).max()
        ),
        "torso_reference_error_max_deg": float(torso_error.max()),
        "board_position_error_mean_m": float(board_pos_error.mean()),
        "board_position_error_final_m": float(board_pos_error[-1]),
        "board_yaw_net_deg": float(board_yaw[-1] - board_yaw[0]),
        "reference_board_yaw_net_deg": float(
            ref_board_yaw[-1] - ref_board_yaw[0]
        ),
        "board_yaw_net_error_deg": float(
            abs((board_yaw[-1] - board_yaw[0]) - (ref_board_yaw[-1] - ref_board_yaw[0]))
        ),
        "landing_both_deck_fraction": float(both_deck[landing].mean()),
        "trailing_both_deck_duration_s": trailing_true_duration(both_deck, dt),
        "any_foot_floor_fraction": float(
            np.any(observations[:, foot_floor] > 0, axis=1).mean()
        ),
        "robot_floor_fraction": float((observations[:, robot_floor] > 0).mean()),
        "invalid_board_floor_fraction": float(invalid_board_floor.mean()),
        # Informational only: this sensor includes legitimate wheel-ground contact.
        "skateboard_floor_fraction": float((observations[:, board_floor] > 0).mean()),
        "hold_torso_height_min_m": float(hold_obs[:, torso_pos[2]].min()),
        "hold_torso_tilt_max_deg": float(
            torso_tilt_deg(hold_obs[:, torso_quat]).max()
        ),
        "hold_both_deck_fraction": float(hold_both_deck.mean()),
        "hold_foot_floor_fraction": float(
            np.any(hold_obs[:, foot_floor] > 0, axis=1).mean()
        ),
        "hold_robot_floor_fraction": float((hold_obs[:, robot_floor] > 0).mean()),
        "hold_invalid_board_floor_fraction": float(hold_invalid_board_floor.mean()),
    }
    gates = {
        "finite": metrics["finite"],
        "torso_height": metrics["torso_height_min_m"] >= 0.65,
        "torso_tilt": metrics["torso_tilt_max_deg"] <= 45.0,
        "board_yaw": metrics["board_yaw_net_error_deg"] <= 25.0,
        "board_position": metrics["board_position_error_mean_m"] <= 0.15,
        "landing_contact": metrics["landing_both_deck_fraction"] >= 0.8,
        "continuous_landing": metrics["trailing_both_deck_duration_s"] >= 0.2,
        "no_foot_floor": metrics["any_foot_floor_fraction"] == 0.0,
        "no_robot_floor": metrics["robot_floor_fraction"] == 0.0,
        "no_invalid_board_floor": metrics["invalid_board_floor_fraction"] == 0.0,
        "stable_hold": (
            metrics["hold_torso_height_min_m"] >= 0.65
            and metrics["hold_torso_tilt_max_deg"] <= 45.0
            and metrics["hold_both_deck_fraction"] >= 0.8
            and metrics["hold_foot_floor_fraction"] == 0.0
            and metrics["hold_robot_floor_fraction"] == 0.0
            and metrics["hold_invalid_board_floor_fraction"] == 0.0
        ),
    }
    ref_path = Path(cfg.task.cfg_ref.motion_path)
    return {
        "passed": bool(all(gates.values())),
        "gates": gates,
        "metrics": metrics,
        "provenance": {
            "run_directory": str(rundir.resolve()),
            "reference": str(ref_path.resolve()),
            "reference_sha256": file_sha256(ref_path),
            "solver_seed": int(cfg.solver.cfg.seed),
            "n_samples": int(cfg.solver.cfg.N_samples),
            "step_knots": int(cfg.task.sim.cfg.step_knots),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rundir", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = evaluate_run(args.rundir, args.reference)
    output = args.output or args.rundir / "evaluation.yaml"
    with output.open("w") as stream:
        yaml.safe_dump(report, stream, sort_keys=False)
    print(yaml.safe_dump(report, sort_keys=False), end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
