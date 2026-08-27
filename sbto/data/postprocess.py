import numpy as np
import numpy.typing as npt
from typing import Dict, Any

from sbto.sim.scene_mj import MjScene
from .constants import *

Array = npt.NDArray[np.float64]

def split_x_traj(
    x_traj: Array,
    mj_scene: MjScene,
    only_pos: bool = False,
    ) -> Dict[str, Array]:
    """
    Split x_traj data into subarrays:
    """
    name2id = {}
    nq = mj_scene.Nq

    # Actuated joints
    act_qposadr = mj_scene.act_qposadr
    act_dofadr = mj_scene.act_dofadr
    name2id[KEY_DOF_POS] = act_qposadr
    if not only_pos:
        name2id[KEY_DOF_V] = act_dofadr + nq

    # Floating base
    if mj_scene.is_floating_base:
        name2id[KEY_ROOT_POS] = mj_scene.base_pos_adr
        name2id[KEY_ROOT_ROT] = mj_scene.base_quat_adr
        if not only_pos:
            name2id[KEY_ROOT_V] = mj_scene.base_v_adr
            name2id[KEY_ROOT_W] = mj_scene.base_w_adr

    # Object
    if mj_scene.is_obj:
        name2id[KEY_OBJECT_POS] = mj_scene.obj_pos_adr
        name2id[KEY_OBJECT_ROT] = mj_scene.obj_quat_adr
        if not only_pos:
            name2id[KEY_OBJECT_V] = mj_scene.obj_v_adr
            name2id[KEY_OBJECT_W] = mj_scene.obj_w_adr

    n_extracted_joints = 0
    for name, id in name2id.items():
        n_extracted_joints += np.sum(np.shape(id))

    if n_extracted_joints > x_traj.shape[-1]:
        raise ValueError(f"Too many extracted joints (got {n_extracted_joints}, reference has {x_traj.shape[-1]})")

    # Extract data
    extracted_data = {}
    n_dim_traj = x_traj.ndim
    for name, id in name2id.items():
        if n_dim_traj == 2:
            id_ = np.atleast_2d(id)
        elif n_dim_traj == 3:
            id_ = id[None, None, :]
        extracted_data[name] = np.take_along_axis(x_traj, id_, axis=-1)

    return extracted_data

def reconstruct_x_traj_from_data_dict(data_dict):
    """
    Reconstruct original trajectory dictionary from split keys.
    """
    x_traj = []
    for k in KEYS_QPOS + KEYS_QVEL:
        if k in data_dict:
            x_traj.append(data_dict[k])
    return np.concatenate(x_traj, axis=-1)

def remove_field_from_data(traj_file_path: str, field: str) -> None:
    """
    Remove field from data.
    """
    file = np.load(traj_file_path)
    data = {k: v for k, v in file.items() if k != "field"}
    np.savez_compressed(
        traj_file_path,
        **data
    )
    print(f"'{field}' data removed from {traj_file_path}")

def remove_obs_from_data(traj_file_path: str,) -> None:
    remove_field_from_data(traj_file_path, KEY_OBS)

def remove_x_from_data(traj_file_path: str,) -> None:
    remove_field_from_data(traj_file_path, KEY_FULL_STATE)
