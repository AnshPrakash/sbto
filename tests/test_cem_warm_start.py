from dataclasses import asdict
from types import SimpleNamespace

import numpy as np

from sbto.solvers.cem import CEM, ConfigCEM
from sbto.utils.hydra import get_warm_start_state_solver


def test_cem_keeps_population_fraction_and_evaluates_anchor():
    solver = CEM(
        2,
        ConfigCEM(
            N_samples=100,
            elite_frac=0.1,
            keep_frac=0.03,
            seed=0,
        ),
    )
    solver.state = solver.init_state(mean=np.array([1.0, 2.0]))

    samples = solver.get_samples().copy()

    assert solver.N_keep == 3
    assert np.array_equal(samples[0], solver.state.mean)

    costs = np.ones(100)
    costs[0] = 0.0
    solver.update(samples, costs)
    next_samples = solver.get_samples()

    assert np.array_equal(next_samples[0], solver.state.best_all)


def test_cem_std_thresholds_use_standard_deviation_units():
    solver = CEM(
        2,
        ConfigCEM(
            N_samples=10,
            elite_frac=0.2,
            sigma0=0.03,
            min_std_collapsed=0.001,
            seed=0,
        ),
    )

    solver.get_samples()

    assert not solver.collapsed_dim.any()
    assert np.isclose(solver.increment_value(), 0.03)


def test_diagonal_cem_samples_and_updates_without_dense_covariance():
    solver = CEM(
        3,
        ConfigCEM(
            N_samples=20,
            elite_frac=0.2,
            diagonal_covariance=True,
            sigma0=0.2,
            seed=0,
        ),
    )
    solver.state = solver.init_state(
        mean=np.array([1.0, 2.0, 3.0]),
        cov=np.diag([0.01, 0.04, 0.09]),
    )

    samples = solver.get_samples().copy()
    solver.update_distrib_param(solver.state, samples[:4])

    assert np.array_equal(samples[0], [1.0, 2.0, 3.0])
    assert np.count_nonzero(
        solver.state.cov - np.diag(np.diag(solver.state.cov))
    ) == 0


def test_finetune_restarts_distribution_from_previous_best(tmp_path):
    solver = CEM(2, ConfigCEM(N_samples=10, elite_frac=0.2, seed=0))
    state = solver.init_state(mean=np.array([1.0, 2.0]))
    state.best_all = np.array([3.0, 4.0])
    state.min_cost_all = 5.0
    np.savez(tmp_path / "solver_state_final.npz", **asdict(state))
    cfg = SimpleNamespace(
        init_knots_from_ref=False,
        init_control_path="",
        warm_start=SimpleNamespace(
            rundir=str(tmp_path), cp_best=False, add_cov_diag=0.0
        ),
    )

    warm_state = get_warm_start_state_solver(cfg, None, None, solver)

    assert np.array_equal(warm_state.mean, state.best_all)
    assert np.isinf(warm_state.min_cost_all)
