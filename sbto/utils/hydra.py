import copy
import glob
import os
from functools import partial

import numpy as np
import yaml
from hydra.utils import instantiate
from omegaconf import OmegaConf

from sbto.data.constants import BEST_TRAJECTORY_FILENAME, KEY_PD_KNOTS, KEY_PD_TARGET
from sbto.data.load import get_final_state_from_rundir
from sbto.data.save import save_results
from sbto.run.optimize import (
    optimize_incremental_opt,
    optimize_mutiple_shooting,
    optimize_single_shooting,
)
from sbto.run.stats import OptimizationStats
from sbto.sim.sim_base import SimRolloutBase
from sbto.solvers.solver_base import SamplingBasedSolver, SolverState
from sbto.tasks.task_base import OCPBase
from sbto.tasks.task_mj_ref import TaskMjRef


def optimize_and_save_data(
    cfg,
    sim: SimRolloutBase,
    task: OCPBase,
    solver: SamplingBasedSolver,
    hydra_rundir: str = "",
    solver_state_0: SolverState | None = None,
    opt_stats: OptimizationStats | None = None,
    ) -> str:

    # Copy initial state
    if solver_state_0:
        solver_state_0 = copy.deepcopy(solver_state_0)
    else:
        solver_state_0 = copy.deepcopy(solver.state)

    # Multiple_shooting
    if cfg.warm_start.multiple_shooting:
        if not isinstance(task, TaskMjRef):
            raise ValueError("Task should be an instance of TaskMjRef (with reference)")
        optimizer_fn = optimize_mutiple_shooting

    # Incremental opt
    elif cfg.warm_start.incremental:
        optimizer_fn = partial(
            optimize_incremental_opt,
            N_max_it_per_knots=cfg.warm_start.N_max_incr,
            min_std_next=cfg.warm_start.min_std_next,
            min_std_final=cfg.warm_start.min_std_final,
        )

    # Single shooting
    else:
        optimizer_fn = optimize_single_shooting
    
    solver_state_final, all_samples, best_samples_it, all_costs, opt_stats = optimizer_fn(
        sim,
        task,
        solver,
        solver_state_0,
        opt_stats,
    )

    rundir = save_results(
        cfg.data_processing.data_dir,
        sim,
        task,
        solver_state_0,
        solver_state_final,
        all_samples,
        best_samples_it,
        all_costs,
        cfg.exp_name,
        cfg.description,
        hydra_rundir,
        cfg.data_processing.save_fig,
        cfg.data_processing.save_video,
        cfg.data_processing.save_samples_costs,
        cfg.data_processing.save_best_samples_it,
        cfg.warm_start.multiple_shooting,
        cfg.data_processing.split_state,
        cfg.data_processing.save_top,
        cfg.data_processing.n_last_it,
        cfg.data_processing.remove_keys,
    )

    opt_stats.save(rundir)

    return rundir, opt_stats

def instantiate_from_cfg(cfg):
    sim = instantiate(cfg.task.sim)
    task = instantiate(cfg.task, sim=sim)
    random = instantiate(cfg.random, sim=sim, seed=cfg.solver.cfg.seed)
    solver = instantiate(cfg.solver, D=sim.Nvars_u)
    return sim, task, solver, random

def get_initial_state_solver_from_ref(sim, task, solver):
    if not isinstance(task, TaskMjRef):
        print("Task has no reference.")
        return None
    qpos_from_ref = task.ref.act_qpos[sim.t_knots, :]
    pd_knots_from_ref = sim.scaling.inverse(qpos_from_ref).reshape(-1)
    solver_state_0 = solver.init_state(mean=pd_knots_from_ref)
    return solver_state_0

def get_initial_state_solver_from_controls(sim, solver, control_path):
    if os.path.isdir(control_path):
        control_path = os.path.join(control_path, f"{BEST_TRAJECTORY_FILENAME}.npz")
    with np.load(control_path) as data:
        if KEY_PD_KNOTS in data:
            controls = np.asarray(data[KEY_PD_KNOTS])
        elif KEY_PD_TARGET in data:
            controls = np.asarray(data[KEY_PD_TARGET])
        else:
            raise ValueError(
                f"{control_path} has neither {KEY_PD_KNOTS!r} nor "
                f"{KEY_PD_TARGET!r}"
            )
    if controls.ndim != 2 or controls.shape[1] != sim.Nu:
        raise ValueError(
            f"Initial controls must have shape (T, {sim.Nu}); got {controls.shape}"
        )
    if len(controls) != sim.Nknots:
        indices = np.rint(np.linspace(0, len(controls) - 1, sim.Nknots)).astype(int)
        controls = controls[indices]
    knots = sim.scaling.inverse(controls).reshape(-1)
    return solver.init_state(mean=knots)

def get_warm_start_state_solver(cfg, sim, task, solver) -> SolverState:
    # Set initial solver state
    solver_state_0 = None
    if cfg.init_knots_from_ref and isinstance(task, TaskMjRef):
        solver_state_0 = get_initial_state_solver_from_ref(sim, task, solver)

    if cfg.init_control_path:
        solver_state_0 = get_initial_state_solver_from_controls(
            sim, solver, cfg.init_control_path
        )

    if cfg.warm_start.rundir and os.path.exists(cfg.warm_start.rundir):
        solver_state_0 = get_final_state_from_rundir(cfg.warm_start.rundir, solver)

        if not cfg.warm_start.cp_best:
            solver_state_0.mean = solver_state_0.best_all.copy()
            solver.reset_min_cost_best(solver_state_0)

        if cfg.warm_start.add_cov_diag > 0.:
            N = solver_state_0.mean.shape[0]
            solver_state_0.cov += cfg.warm_start.add_cov_diag * np.eye(N)

    return solver_state_0

def set_cfg_warm_start(cfg):
    cfg_ws = copy.deepcopy(cfg)
    WARM_START_MULTIPLE_SHOOTING = "ws_ms"
    WARM_START_INCREMENTAL = "ws_incr"

    # Update description
    sep = "_" if cfg.description else ""
    if cfg_ws.warm_start.incremental:
        cfg_ws.description += sep + WARM_START_INCREMENTAL
    
    elif cfg_ws.warm_start.multiple_shooting:
        cfg_ws.description += sep + WARM_START_MULTIPLE_SHOOTING
    return cfg_ws

def get_optimization_stats_warm_start(cfg) -> OptimizationStats | None:
    rundir = cfg.warm_start.rundir
    if rundir and os.path.exists(rundir):
        opt_stats = OptimizationStats.load(rundir)
    else:
        opt_stats = None
    return opt_stats

def load_yaml(yaml_path):
    d = {}
    if os.path.exists(yaml_path):
        with open(yaml_path, "r") as f:
            d = yaml.safe_load(f)
    return d

def save_yaml(yaml_path, data):
    if os.path.exists(yaml_path):
        with open(yaml_path, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)

def update_cfg_from_warm_start(cfg, hydra_rundir: str):
    rundir = cfg.warm_start.rundir
    
    if rundir and os.path.exists(rundir):
        
        # update config params from warm_start config
        cfg_paths = glob.glob(
            f"{rundir}/**/config.yaml",
            include_hidden=True,
            recursive=True
        )
        if len(cfg_paths) > 0:
            cfg_dict = load_yaml(cfg_paths[0])
            # Fall back to the old run only when the current reference is unavailable.
            cfg_warm_start = OmegaConf.create(cfg_dict)
            current_motion = os.path.expanduser(cfg.task.cfg_ref.motion_path)
            warm_motion = os.path.expanduser(cfg_warm_start.task.cfg_ref.motion_path)
            if not os.path.exists(current_motion) and os.path.exists(warm_motion):
                cfg.task.cfg_ref.motion_path = cfg_warm_start.task.cfg_ref.motion_path
            # cfg.task.sim.cfg.Nknots = cfg_warm_start.task.sim.cfg.Nknots
            # save yaml

            current_cfg_path = glob.glob(
                f"{hydra_rundir}/**/config.yaml",
                include_hidden=True,
                recursive=True
            )[0]
            save_yaml(current_cfg_path, OmegaConf.to_object(cfg))
