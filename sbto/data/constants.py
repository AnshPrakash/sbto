DATA_DIR = "./datasets"
TRAJ_FILENAME = "time_x_u_traj"
CONFIG_FILENAME = "config"
ROLLOUT_FILENAME = "rollout_time_x_u_obs_traj"
SOLVER_STATES_DIR = "./solver_states"
ALL_SAMPLES_COSTS_FILENAME = "samples_costs"
BEST_SAMPLES_IT_FILENAME = "best_samples_it"
SOLVER_STATE_NAME = "solver_state"
INITIAL_SOLVER_STATE_SUFFIX = "0"
FINAL_SOLVER_STATE_SUFFIX = "final"
HYDRA_CFG = ".hydra"
MJ_MODEL_NAME = "mj_model"
BEST_TRAJECTORY_FILENAME = "best_trajectory"
TOP_TRAJECTORIES_FILENAME = "top_trajectories"
BEST_TRAJECTORY_RAND_FILENAME = "best_trajectory_rand"

# qpos keys
KEY_ROOT_POS      = "root_pos"
KEY_ROOT_ROT      = "root_rot"
KEY_DOF_POS       = "dof_pos"
KEY_OBJECT_POS    = "object_pos"
KEY_OBJECT_ROT    = "object_rot"

# qvel keys
KEY_ROOT_V        = "root_lin_vel"
KEY_ROOT_W        = "root_ang_vel"
KEY_DOF_V         = "dof_vel"
KEY_OBJECT_V      = "object_lin_vel"
KEY_OBJECT_W      = "object_ang_vel"

KEYS_QPOS = [
    KEY_ROOT_POS,
    KEY_ROOT_ROT,
    KEY_DOF_POS,
    KEY_OBJECT_POS,
    KEY_OBJECT_ROT,
]

KEYS_QVEL = [
    KEY_ROOT_V,
    KEY_ROOT_W,
    KEY_DOF_V,
    KEY_OBJECT_V,
    KEY_OBJECT_W,
]

KEY_FULL_STATE = "x"
KEY_TIME = "time"
KEY_OBS = "obs"
KEY_PD_TARGET  = "dof_pd_target"
KEY_PD_TARGET_CANDIDATES = "dof_pd_target_candidates"
KEY_PD_KNOTS = "dof_pd_knots"
KEY_COST  = "cost"
KEY_STEP_KNOTS  = "step_knots"
