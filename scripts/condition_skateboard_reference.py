"""Prepend a physically checked on-board stance to an SBTO skateboard reference."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import mujoco
import numpy as np
import yaml
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from mujoco import rollout

from sbto.data.constants import KEY_PD_KNOTS, KEY_PD_TARGET, KEY_STEP_KNOTS
from sbto.data.utils import load_best_trajectory_from_rundir
from sbto.evaluation.skateboard import torso_tilt_deg

NOMINAL_JOINT_POSITIONS = {
    "left_hip_pitch": -0.1,
    "right_hip_pitch": -0.1,
    "knee": 0.3,
    "ankle_pitch": -0.2,
    "left_shoulder_pitch": 0.3,
    "right_shoulder_pitch": 0.3,
    "left_shoulder_roll": 0.25,
    "right_shoulder_roll": -0.25,
    "elbow": 0.97,
    "left_wrist_roll": 0.15,
    "right_wrist_roll": -0.15,
}


def load_skateboard_sim():
    config_dir = str((Path(__file__).parents[1] / "sbto" / "conf").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="config", overrides=["task=g1/skateboard_ref"])
    return instantiate(cfg.task.sim)


def matching_nominal_position(joint_name: str) -> float:
    matches = [
        (len(pattern), value)
        for pattern, value in NOMINAL_JOINT_POSITIONS.items()
        if pattern in joint_name
    ]
    return max(matches)[1] if matches else 0.0


def normalized_lerp(a: np.ndarray, b: np.ndarray, weight: float) -> np.ndarray:
    if np.dot(a, b) < 0.0:
        b = -b
    result = (1.0 - weight) * a + weight * b
    return result / np.linalg.norm(result)


def interpolate_qpos(
    model: mujoco.MjModel, start: np.ndarray, end: np.ndarray, weight: float
) -> np.ndarray:
    result = (1.0 - weight) * start + weight * end
    free_joints = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    for joint_id in free_joints:
        address = model.jnt_qposadr[joint_id] + 3
        result[address : address + 4] = normalized_lerp(
            start[address : address + 4], end[address : address + 4], weight
        )
    return result


def build_conditioned_reference(
    sim,
    source_qpos: np.ndarray,
    source_qvel: np.ndarray,
    fps: float,
    static_duration: float,
    blend_duration: float,
    terminal_hold_duration: float,
    root_height_above_board: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    model = sim.mj_scene.mj_model
    if source_qpos.ndim != 2 or source_qpos.shape[1] != model.nq:
        raise ValueError(f"qpos must have shape (T, {model.nq}); got {source_qpos.shape}")
    if source_qvel.shape != (len(source_qpos), model.nv):
        raise ValueError(
            f"qvel must have shape ({len(source_qpos)}, {model.nv}); "
            f"got {source_qvel.shape}"
        )
    if not np.isfinite(source_qpos).all() or fps <= 0.0:
        raise ValueError("Reference must contain finite qpos and a positive fps")

    stance = model.qpos0.copy()
    stance[:2] = source_qpos[0, sim.mj_scene.obj_pos_adr[:2]]
    stance[2] = source_qpos[0, sim.mj_scene.obj_pos_adr[2]] + root_height_above_board
    stance[3:7] = source_qpos[0, 3:7]
    for joint_id, qpos_address in zip(
        sim.mj_scene.act_joint_ids, sim.mj_scene.act_qposadr
    ):
        stance[qpos_address] = matching_nominal_position(
            model.joint(int(joint_id)).name
        )
    object_start = sim.mj_scene.obj_pos_adr[0]
    stance[object_start:] = source_qpos[0, object_start:]

    if (static_duration == 0.0) != (blend_duration == 0.0):
        raise ValueError("static and blend durations must either both be zero or positive")
    use_leadin = static_duration > 0.0
    static_frames = max(2, round(static_duration * fps)) if use_leadin else 0
    blend_frames = max(2, round(blend_duration * fps)) if use_leadin else 0
    terminal_hold_frames = max(2, round(terminal_hold_duration * fps))
    if use_leadin:
        qpos = [stance.copy() for _ in range(static_frames)]
        for phase in np.linspace(0.0, 1.0, blend_frames + 1)[1:]:
            smooth_phase = phase * phase * (3.0 - 2.0 * phase)
            qpos.append(interpolate_qpos(model, stance, source_qpos[0], smooth_phase))
        qpos.extend(source_qpos[1:])
    else:
        qpos = list(source_qpos)
    qpos.extend(source_qpos[-1].copy() for _ in range(terminal_hold_frames))
    qpos = np.asarray(qpos)

    qvel = np.zeros((len(qpos), model.nv))
    if use_leadin:
        dt = 1.0 / fps
        for index in range(len(qpos) - 1):
            mujoco.mj_differentiatePos(
                model, qvel[index], dt, qpos[index], qpos[index + 1]
            )
        qvel[-1] = qvel[-2]
    else:
        qvel[: len(source_qvel)] = source_qvel
    return qpos, qvel, static_frames


def sensor_indices(model: mujoco.MjModel, names: tuple[str, ...] | str) -> np.ndarray:
    names = (names,) if isinstance(names, str) else names
    indices = []
    for name in names:
        sensor = model.sensor(name)
        indices.extend(range(sensor.adr[0], sensor.adr[0] + sensor.dim[0]))
    return np.asarray(indices)


def validate_stance(sim, qpos: np.ndarray, static_frames: int) -> dict:
    model = sim.mj_scene.mj_model
    x0 = np.concatenate((qpos[0], np.zeros(model.nv)))
    control = np.repeat(
        qpos[0, sim.mj_scene.act_qposadr][None, None], static_frames, axis=1
    )
    _, observations = rollout.rollout(
        model, mujoco.MjData(model), np.concatenate(([0.0], x0))[None], control
    )
    observations = observations[0]
    torso_pos = sensor_indices(model, "global_pos_torso")
    torso_quat = sensor_indices(model, "orientation_torso")
    deck = sensor_indices(model, ("left_foot_deck", "right_foot_deck"))
    foot_floor = sensor_indices(
        model,
        (
            "left_foot1_floor",
            "left_foot2_floor",
            "left_foot3_floor",
            "right_foot1_floor",
            "right_foot2_floor",
            "right_foot3_floor",
        ),
    )
    robot_floor = sensor_indices(model, "robot_floor")
    both_deck = np.all(observations[:, deck] > 0, axis=1)
    metrics = {
        "torso_height_min_m": float(observations[:, torso_pos[2]].min()),
        "torso_tilt_max_deg": float(
            torso_tilt_deg(observations[:, torso_quat]).max()
        ),
        "both_deck_fraction": float(both_deck.mean()),
        "foot_floor_fraction": float(
            np.any(observations[:, foot_floor] > 0, axis=1).mean()
        ),
        "robot_floor_fraction": float((observations[:, robot_floor] > 0).mean()),
    }
    metrics["passed"] = bool(
        metrics["torso_height_min_m"] >= 0.65
        and metrics["torso_tilt_max_deg"] <= 45.0
        and metrics["both_deck_fraction"] == 1.0
        and metrics["foot_floor_fraction"] == 0.0
        and metrics["robot_floor_fraction"] == 0.0
    )
    return metrics


def build_conditioned_controls(
    sim,
    stance_qpos: np.ndarray,
    source_controls: np.ndarray,
    static_duration: float,
    blend_duration: float,
    terminal_hold_duration: float,
) -> np.ndarray:
    if source_controls.ndim != 2 or source_controls.shape[1] != sim.Nu:
        raise ValueError(
            f"Warm-start controls must have shape (T, {sim.Nu}); "
            f"got {source_controls.shape}"
        )
    dt = sim.mj_scene.dt
    if (static_duration == 0.0) != (blend_duration == 0.0):
        raise ValueError("static and blend durations must either both be zero or positive")
    use_leadin = static_duration > 0.0
    static_steps = max(2, round(static_duration / dt)) if use_leadin else 0
    blend_steps = max(2, round(blend_duration / dt)) if use_leadin else 0
    hold_steps = max(2, round(terminal_hold_duration / dt))
    stance_control = stance_qpos[sim.mj_scene.act_qposadr]
    controls = [stance_control.copy() for _ in range(static_steps)]
    if use_leadin:
        for phase in np.linspace(0.0, 1.0, blend_steps + 1)[1:]:
            smooth_phase = phase * phase * (3.0 - 2.0 * phase)
            controls.append(
                (1.0 - smooth_phase) * stance_control
                + smooth_phase * source_controls[0]
            )
    controls.extend(source_controls)
    controls.extend(source_controls[-1].copy() for _ in range(hold_steps))
    return np.asarray(controls)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def condition(
    source: Path,
    destination: Path,
    static_duration: float = 0.3,
    blend_duration: float = 0.3,
    terminal_hold_duration: float = 0.5,
    root_height_above_board: float = 0.784,
    warm_start_run: Path | None = None,
    force: bool = False,
) -> dict:
    if destination.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite {destination}; pass --force")
    sim = load_skateboard_sim()
    with np.load(source) as motion:
        qpos, qvel, static_frames = build_conditioned_reference(
            sim,
            np.asarray(motion["qpos"]),
            np.asarray(motion["qvel"]),
            float(motion["fps"]),
            static_duration,
            blend_duration,
            terminal_hold_duration,
            root_height_above_board,
        )
        fps = float(motion["fps"])
    validation = (
        validate_stance(sim, qpos, static_frames)
        if static_frames
        else {"passed": True, "skipped": True}
    )
    if not validation["passed"]:
        raise RuntimeError(f"Conditioned stance failed physical validation: {validation}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, qpos=qpos, qvel=qvel, fps=fps)
    report = {
        "input": str(source.resolve()),
        "input_sha256": sha256(source),
        "output": str(destination.resolve()),
        "output_sha256": sha256(destination),
        "frames": len(qpos),
        "fps": fps,
        "static_duration_s": static_duration,
        "blend_duration_s": blend_duration,
        "terminal_hold_duration_s": terminal_hold_duration,
        "root_height_above_board_m": root_height_above_board,
        "stance_validation": validation,
    }
    if warm_start_run is not None:
        warm_start_data = load_best_trajectory_from_rundir(str(warm_start_run))
        source_controls = np.asarray(warm_start_data[KEY_PD_TARGET])
        controls = build_conditioned_controls(
            sim,
            qpos[0],
            source_controls,
            static_duration,
            blend_duration,
            terminal_hold_duration,
        )
        controls_path = destination.with_name(
            f"{destination.stem}-initial-controls.npz"
        )
        payload = {KEY_PD_TARGET: controls}
        if KEY_STEP_KNOTS in warm_start_data:
            source_steps = np.atleast_1d(
                np.asarray(warm_start_data[KEY_STEP_KNOTS], dtype=int)
            )
            if len(source_steps) >= 2:
                source_knots = source_controls[
                    np.clip(source_steps, 0, len(source_controls) - 1)
                ]
                interval = float(np.mean(np.diff(source_steps))) * sim.mj_scene.dt
                lead_knots = round((static_duration + blend_duration) / interval)
                hold_knots = round(terminal_hold_duration / interval)
                if lead_knots:
                    lead_end = round(
                        (static_duration + blend_duration) / sim.mj_scene.dt
                    )
                    lead_indices = np.rint(
                        np.linspace(0, lead_end - 1, lead_knots)
                    ).astype(int)
                    source_knots = np.vstack((controls[lead_indices], source_knots))
                payload[KEY_PD_KNOTS] = np.vstack(
                    (source_knots, np.repeat(source_knots[-1:], hold_knots, axis=0))
                )
        np.savez_compressed(controls_path, **payload)
        report["initial_controls"] = str(controls_path.resolve())
        report["initial_controls_sha256"] = sha256(controls_path)
    with destination.with_suffix(".yaml").open("w") as stream:
        yaml.safe_dump(report, stream, sort_keys=False)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--static-duration", type=float, default=0.3)
    parser.add_argument("--blend-duration", type=float, default=0.3)
    parser.add_argument("--terminal-hold-duration", type=float, default=0.5)
    parser.add_argument("--root-height-above-board", type=float, default=0.784)
    parser.add_argument("--warm-start-run", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    report = condition(
        args.source,
        args.destination,
        args.static_duration,
        args.blend_duration,
        args.terminal_hold_duration,
        args.root_height_above_board,
        args.warm_start_run,
        args.force,
    )
    print(yaml.safe_dump(report, sort_keys=False), end="")


if __name__ == "__main__":
    main()
