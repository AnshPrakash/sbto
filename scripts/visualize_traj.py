import argparse

import mujoco
from hydra.utils import instantiate

from sbto.data.constants import *
from sbto.data.utils import get_config_from_rundir, load_best_trajectory_from_rundir
from sbto.main import instantiate_from_cfg
from sbto.utils.viewer import visualize_trajectory, visualize_trajectory_with_reference


def main(rundir: str, with_ref: bool = True, reference_path: str | None = None):

    cfg = get_config_from_rundir(rundir)
    data = load_best_trajectory_from_rundir(rundir)

    if with_ref:
        if reference_path:
            cfg.task.cfg_ref.motion_path = reference_path
        sim, task, _, _ = instantiate_from_cfg(cfg)
    else:
        sim = instantiate(cfg.task.sim)
    mj_model = sim.mj_scene.mj_model
    mj_data = mujoco.MjData(mj_model)

    if with_ref:
        visualize_trajectory_with_reference(
            mj_model, mj_data, task.ref.time, data[KEY_FULL_STATE], task.ref.x
        )
    else:
        visualize_trajectory(
            mj_model, mj_data, data[KEY_TIME], data[KEY_FULL_STATE]
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize best trajectory from a run directory."
    )

    parser.add_argument(
        "rundir",
        type=str,
        help="Path to run directory containing config and trajectory data.",
    )

    parser.add_argument(
        "--no-ref",
        action="store_true",
        help="Disable reference trajectory visualization.",
    )
    parser.add_argument(
        "--reference",
        help="Override the reference path stored in the run configuration.",
    )

    args = parser.parse_args()

    main(
        rundir=args.rundir,
        with_ref=not args.no_ref,
        reference_path=args.reference,
    )
