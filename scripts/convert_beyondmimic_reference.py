"""Convert a trusted BeyondMimic motion archive to SBTO model order."""

import argparse
from pathlib import Path

import mujoco
import numpy as np
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate


def load_skateboard_model():
    config_dir = str((Path(__file__).parents[1] / "sbto" / "conf").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="config", overrides=["task=g1/skateboard_ref"])
    return instantiate(cfg.task.sim).mj_scene.mj_model


def convert(source: Path, destination: Path) -> None:
    model = load_skateboard_model()
    motion = np.load(source, allow_pickle=True)
    nframes = len(motion["robot_root_pos_w"])
    qpos = np.tile(model.qpos0, (nframes, 1))
    qvel = np.zeros((nframes, model.nv))

    def copy_free_joint(joint_id, prefix):
        qadr = model.jnt_qposadr[joint_id]
        dadr = model.jnt_dofadr[joint_id]
        qpos[:, qadr : qadr + 3] = motion[f"{prefix}_root_pos_w"]
        qpos[:, qadr + 3 : qadr + 7] = motion[f"{prefix}_root_quat_w"]
        qvel[:, dadr : dadr + 3] = motion[f"{prefix}_root_lin_vel_w"]
        qvel[:, dadr + 3 : dadr + 6] = motion[f"{prefix}_root_ang_vel_w"]

    free_joints = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free_joints) != 2:
        raise ValueError(f"Expected robot and skateboard free joints; got {len(free_joints)}")
    copy_free_joint(free_joints[0], "robot")
    copy_free_joint(free_joints[1], "object")

    for prefix in ("robot", "object"):
        names = motion[f"{prefix}_joint_names"].tolist()
        positions = motion[f"{prefix}_joint_pos"]
        velocities = motion[f"{prefix}_joint_vel"]
        if positions.shape != velocities.shape or positions.shape != (nframes, len(names)):
            raise ValueError(f"Invalid {prefix} joint array shapes")
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate {prefix} joint names")
        for column, name in enumerate(names):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"Joint {name!r} is absent from the SBTO model")
            qpos[:, model.jnt_qposadr[joint_id]] = positions[:, column]
            qvel[:, model.jnt_dofadr[joint_id]] = velocities[:, column]

    if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
        raise ValueError("Converted reference contains non-finite values")
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, qpos=qpos, qvel=qvel, fps=float(motion["fps"]))
    print(f"Wrote {nframes} frames: qpos {qpos.shape}, qvel {qvel.shape} -> {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    convert(args.source, args.destination)
