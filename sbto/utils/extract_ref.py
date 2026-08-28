import mujoco
import numpy as np
from scipy.interpolate import interp1d

from sbto.data.constants import *
from sbto.data.postprocess import split_x_traj
from sbto.sim.scene_mj import MjScene
from sbto.utils.finite_diff import (
    finite_diff_qpos_traj_high_order,
    finite_diff_quat_traj,
)


def normalize_quat(q: np.ndarray) -> np.ndarray:
    """Normalize quaternion array [T,4]."""
    return q / np.linalg.norm(q, axis=-1, keepdims=True)

def quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [x,y,z,w] -> [w,x,y,z]."""
    return np.column_stack([q[:, 3], q[:, :3]])

def flip_quat_pos_in_traj(free_joint_traj: np.ndarray) -> np.ndarray:
    """Convert free joint [quat, pos] -> [pos, quat]."""
    free_joint_traj_flipped = np.empty_like(free_joint_traj)
    free_joint_traj_flipped[:, :3] = free_joint_traj[:, -3:]
    free_joint_traj_flipped[:, 3:] = free_joint_traj[:, :-3]
    return free_joint_traj_flipped

def compute_time_array(fps: float, N: int) -> np.ndarray:
    return np.arange(N) / fps

def load_npz_reference(path: str) -> dict[str, np.ndarray]:
    """
    Load model-ordered qpos and optional qvel from an NPZ reference.
    """
    file = np.load(path, mmap_mode="r")
    data = {"qpos": file["qpos"], "fps": float(file["fps"])}
    if "qvel" in file:
        data["qvel"] = file["qvel"]
    return data

def make_quaternions_continuous(quat_traj: np.ndarray):
    """
    Ensures that quaternions are continuous with sign flip.
    """
    quat_dot_prod = np.sum(quat_traj[:-1, :] * quat_traj[1:, :], axis=-1)
    sign_flip = np.argwhere(quat_dot_prod < 0) + 1
    for id in sign_flip:
        quat_traj[np.squeeze(id):, :] *= -1
    return quat_traj

def interpolate_trajectory(
    values: np.ndarray, time: np.ndarray, t_new: np.ndarray, is_quat=False
):
    """Generic interpolation for batched arrays."""
    if is_quat:
        values = make_quaternions_continuous(values)
        interp = interp1d(time, values, axis=0, copy=False, assume_sorted=True)
        return normalize_quat(interp(t_new))
    else:
        interp = interp1d(time, values, kind="cubic", axis=0, copy=False, assume_sorted=True)
        return interp(t_new)

class ReferenceMotion:
    """
    Clean, minimal reference motion loader from NPZ.
    - Keeps only qpos + time + fps in data dict.
    - Everything else is provided via properties:
        root_pos, root_rot, dof_pos, etc.
    """
    def __init__(
        self,
        mj_scene: MjScene,
        ref_motion_path: str,
        t0: float = 0.0,
        t_end: float = 0.0,
        speedup: float = 1.0,
        z_offset: float = 0.0,
        flip_quat_pos: bool = True,
        quat_wxyz: bool = True,
    ):
        self.mj_scene = mj_scene
        self.dt = self.mj_scene.dt
        self.sensor_data = {}

        # Load base data
        base = load_npz_reference(ref_motion_path)
        self.fps = base["fps"] * speedup
        self._qpos = np.asarray(base["qpos"], dtype=np.float64).copy()
        self._qvel = (
            np.asarray(base["qvel"], dtype=np.float64).copy() * speedup
            if "qvel" in base
            else None
        )
        if self._qpos.ndim != 2 or self._qpos.shape[1] != self.mj_scene.Nq:
            raise ValueError(
                f"Reference qpos must have shape (T, {self.mj_scene.Nq}); "
                f"got {self._qpos.shape}."
            )
        if self._qvel is not None and self._qvel.shape != (
            len(self._qpos),
            self.mj_scene.Nv,
        ):
            raise ValueError(
                f"Reference qvel must have shape (T, {self.mj_scene.Nv}); "
                f"got {self._qvel.shape}."
            )

        # Fix quaterion format
        if self.mj_scene.is_floating_base:
            base_qpos = np.concatenate((
                self.mj_scene.base_pos_adr,
                self.mj_scene.base_quat_adr
                ))
            if flip_quat_pos:
                self._qpos[:, base_qpos] = flip_quat_pos_in_traj(self._qpos[:, base_qpos])
            if not quat_wxyz:
                self._qpos[:, base_qpos] = quat_xyzw_to_wxyz(self._qpos[:, base_qpos])

        if self.mj_scene.is_obj:
            obj_qpos = self.mj_scene.obj_qpos_adr
            if flip_quat_pos:
                self._qpos[:, obj_qpos] = flip_quat_pos_in_traj(self._qpos[:, obj_qpos])
            if not quat_wxyz:
                self._qpos[:, obj_qpos] = quat_xyzw_to_wxyz(self._qpos[:, obj_qpos])

        self.time = compute_time_array(self.fps, len(self._qpos))
        self.trim_traj(t0, t_end)
        self.apply_z_offset(z_offset)
        self._qpos_dict = split_x_traj(self._qpos, self.mj_scene, only_pos=True)
        source_time = self.time.copy()
        self._qpos_dict = self.interpolate_to_mj_dt(self._qpos_dict)
        if self._qvel is None:
            self._vel_dict = self.compute_velocities(self._qpos_dict)
        else:
            if not np.array_equal(source_time, self.time):
                self._qvel = interpolate_trajectory(
                    self._qvel, source_time, self.time
                )
            self._vel_dict = self.split_velocities(self._qvel)
        self.x = self.concatenate_full_state(self._qpos_dict, self._vel_dict)

    def trim_traj(self, t0: float, t_end: float):
        """Trim trajectory so that new time starts at t0."""
        if t0 <= 0 and t_end <=0:
            return
        
        if t_end <= t0:
            return
        
        if t0 > 0:
            idx = np.searchsorted(self.time, t0)
            self._qpos = self._qpos[idx:]
            if self._qvel is not None:
                self._qvel = self._qvel[idx:]
            self.time = self.time[idx:] - self.time[idx]

        if t_end > 0:
            idx = np.searchsorted(self.time, t_end)
            self._qpos = self._qpos[:idx]
            if self._qvel is not None:
                self._qvel = self._qvel[:idx]
            self.time = self.time[:idx]

    def apply_z_offset(self, z_offset):
        if z_offset != 0:
            self._qpos[:, 2] -= z_offset  # root_pos[2]
            if self.mj_scene.is_obj:
                obj_qpos_z = self.mj_scene.obj_pos_adr[-1]
                self._qpos[:, obj_qpos_z] -= z_offset  # object_pos[2]

    def interpolate_to_mj_dt(self, qpos_dict):
        dt_in = 1.0 / self.fps

        if abs(self.dt - dt_in) < 1e-4:
            return qpos_dict

        t_new = np.arange(0, self.time[-1] + self.dt / 2, self.dt)
        qpos_dict_interp = {}
        for k, v in qpos_dict.items():
            is_quat = "rot" in k
            qpos_dict_interp[k] = interpolate_trajectory(v, self.time, t_new, is_quat=is_quat)
        
        self.time = t_new
        return qpos_dict_interp

    def compute_velocities(self, qpos_dict: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Compute velocities for root/dof/object segments."""
        out = {}

        # Root
        if KEY_ROOT_POS in qpos_dict:
            out[KEY_ROOT_V] = finite_diff_qpos_traj_high_order(qpos_dict[KEY_ROOT_POS], self.dt)
            out[KEY_ROOT_W] = finite_diff_quat_traj(qpos_dict[KEY_ROOT_ROT], self.dt)

        # DOF
        out[KEY_DOF_V] = finite_diff_qpos_traj_high_order(qpos_dict[KEY_DOF_POS], self.dt)

        # Object (optional)
        if KEY_OBJECT_POS in qpos_dict:
            out[KEY_OBJECT_V] = finite_diff_qpos_traj_high_order(qpos_dict[KEY_OBJECT_POS], self.dt)
            out[KEY_OBJECT_W] = finite_diff_quat_traj(qpos_dict[KEY_OBJECT_ROT], self.dt)

        return out

    def split_velocities(self, qvel: np.ndarray) -> dict[str, np.ndarray]:
        """Split model-ordered qvel into the fields used by the task."""
        out = {KEY_DOF_V: qvel[:, self.mj_scene.act_vel_adr - self.mj_scene.Nq]}
        if self.mj_scene.is_floating_base:
            out[KEY_ROOT_V] = qvel[:, self.mj_scene.base_v_adr - self.mj_scene.Nq]
            out[KEY_ROOT_W] = qvel[:, self.mj_scene.base_w_adr - self.mj_scene.Nq]
        if self.mj_scene.is_obj:
            out[KEY_OBJECT_V] = qvel[:, self.mj_scene.obj_v_adr - self.mj_scene.Nq]
            out[KEY_OBJECT_W] = qvel[:, self.mj_scene.obj_w_adr - self.mj_scene.Nq]
        return out

    def concatenate_full_state(self, qpos_dict, vel_dict) -> np.ndarray:
        x = np.zeros((len(self.time), self.mj_scene.Nx))
        x[:, :self.mj_scene.Nq] = self.mj_scene.mj_model.qpos0

        qpos_fields = {
            KEY_DOF_POS: self.mj_scene.act_qposadr,
            KEY_ROOT_POS: self.mj_scene.base_pos_adr,
            KEY_ROOT_ROT: self.mj_scene.base_quat_adr,
            KEY_OBJECT_POS: self.mj_scene.obj_pos_adr,
            KEY_OBJECT_ROT: self.mj_scene.obj_quat_adr,
        }
        qvel_fields = {
            KEY_DOF_V: self.mj_scene.act_vel_adr,
            KEY_ROOT_V: self.mj_scene.base_v_adr,
            KEY_ROOT_W: self.mj_scene.base_w_adr,
            KEY_OBJECT_V: self.mj_scene.obj_v_adr,
            KEY_OBJECT_W: self.mj_scene.obj_w_adr,
        }
        used_qpos = []
        for key, addresses in qpos_fields.items():
            if key in qpos_dict:
                x[:, addresses] = qpos_dict[key]
                used_qpos.extend(addresses)
        for key, addresses in qvel_fields.items():
            if key in vel_dict:
                x[:, addresses] = vel_dict[key]

        passive_qpos = np.setdiff1d(
            np.arange(self.mj_scene.Nq), np.asarray(used_qpos, dtype=int)
        )
        if passive_qpos.size:
            passive_values = self._qpos[:, passive_qpos]
            if self._qvel is None and not np.allclose(
                passive_values, passive_values[0], atol=1e-8
            ):
                raise ValueError(
                    "Reference contains moving passive joints; provide model-ordered "
                    "qvel to preserve them."
                )
            if len(passive_values) == len(self.time):
                x[:, passive_qpos] = passive_values
            else:
                x[:, passive_qpos] = interpolate_trajectory(
                    passive_values,
                    compute_time_array(self.fps, len(passive_values)),
                    self.time,
                )

        if self._qvel is not None:
            x[:, self.mj_scene.Nq:] = self._qvel

        return x

    def compute_sensor_data(self, sensor_names: list[str]):
        """
        Extracts sensor values for each timestep along the trajectory.
        The results are stored as attributes:
            self.<sensor_name>
        
        Sensor trajectory shape:
            [T, sensor_dim]
        """
        model = self.mj_scene.mj_model
        data = mujoco.MjData(model)
        T = len(self.time)

        def get_sid(sensor_name: str):
            return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)

        sensor_name2adr = {
            sensor_name : model.sensor_adr[get_sid(sensor_name)]
            for sensor_name in sensor_names
        }
        sensor_name2dim = {
            sensor_name : model.sensor_dim[get_sid(sensor_name)]
            for sensor_name in sensor_names
        }
        self.sensor_data = {
            sensor_name : np.empty((T, sensor_name2dim[sensor_name]))
            for sensor_name in sensor_names
        }

        for t in range(T):
            data.qpos[:] =  self.x[t, :self.mj_scene.Nq]
            data.qvel[:] =  self.x[t, self.mj_scene.Nq:]
            mujoco.mj_forward(model, data)

            for sensor_name in sensor_names:
                adr = sensor_name2adr[sensor_name]
                dim = sensor_name2dim[sensor_name]
                self.sensor_data[sensor_name][t] = data.sensordata[adr:adr + dim]

    @property
    def T(self): return len(self.time)
    
    @property
    def x0(self): return self.x[0]

    @property
    def root_rot(self): return self._qpos_dict.get(KEY_ROOT_ROT)

    @property
    def root_pos(self): return self._qpos_dict.get(KEY_ROOT_POS)

    @property
    def dof_pos(self): return self._qpos_dict[KEY_DOF_POS]

    @property
    def object_pos(self): return self._qpos_dict.get(KEY_OBJECT_POS)

    @property
    def object_rot(self): return self._qpos_dict.get(KEY_OBJECT_ROT)

    @property
    def root_v(self): return self._vel_dict.get(KEY_ROOT_V)

    @property
    def root_w(self): return self._vel_dict.get(KEY_ROOT_W)

    @property
    def dof_v(self): return self._vel_dict[KEY_DOF_V]

    @property
    def object_v(self): return self._vel_dict.get(KEY_OBJECT_V)

    @property
    def object_w(self): return self._vel_dict.get(KEY_OBJECT_W)

    @property
    def act_qpos(self):
        return self.dof_pos

    @property
    def act_qpos0(self):
        return self.dof_pos[0]
    
    @property
    def act_qpos_range(self):
        return self.act_qpos.min(axis=0), self.act_qpos.max(axis=0)

    @property
    def act_qpos_mean(self):
        return np.mean(self.act_qpos, axis=0)
