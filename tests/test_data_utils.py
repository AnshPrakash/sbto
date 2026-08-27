import numpy as np

from sbto.data.constants import BEST_TRAJECTORY_FILENAME, KEY_FULL_STATE
from sbto.data.utils import load_best_trajectory_from_rundir


def test_loader_preserves_saved_full_state(tmp_path):
    full_state = np.arange(2 * 96).reshape(2, 96)
    split_state = {
        "root_pos": np.zeros((2, 3)),
        "root_rot": np.zeros((2, 4)),
        "dof_pos": np.zeros((2, 29)),
        "object_pos": np.zeros((2, 3)),
        "object_rot": np.zeros((2, 4)),
        "root_lin_vel": np.zeros((2, 3)),
        "root_ang_vel": np.zeros((2, 3)),
        "dof_vel": np.zeros((2, 29)),
        "object_lin_vel": np.zeros((2, 3)),
        "object_ang_vel": np.zeros((2, 3)),
    }
    np.savez(
        tmp_path / f"{BEST_TRAJECTORY_FILENAME}.npz",
        **{KEY_FULL_STATE: full_state},
        **split_state,
    )

    loaded = load_best_trajectory_from_rundir(tmp_path)

    np.testing.assert_array_equal(loaded[KEY_FULL_STATE], full_state)
