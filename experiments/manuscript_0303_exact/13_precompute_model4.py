#!/usr/bin/env python3
from pathlib import Path

import numpy as np

from neuromaps import NeuroMapManuscriptSubnets
from systems.vdp_mod2 import vdp_mod2_rk4, vdp_mod2_right_part
from utils import (
    grid_of_amplitude_basin_over_initial_state,
    grid_of_fixed_point_probability_over_params,
    grid_scan_neuromap_nearest_fixed_point,
)

ARTIFACTS_DIR = Path("experiments/manuscript_0303_exact/artifacts/model4")
RESULTS_DIR = ARTIFACTS_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DT = 0.01
LAM = -0.5
MU = 3.0


def main() -> None:
    model_path = ARTIFACTS_DIR / "checkpoints_subnets" / "model.ckpt"
    if not model_path.is_file():
        raise FileNotFoundError("Run 12_train_model4_subnets.py first.")

    model = NeuroMapManuscriptSubnets.load(str(model_path), device="cpu")

    x_grid = np.linspace(-3.3, 3.3, 100)
    y_grid = np.linspace(-29.0, 29.0, 100)
    base_state = np.zeros(2, dtype=float)
    p_fixed = [LAM, MU]

    Z_ode, _ = grid_of_amplitude_basin_over_initial_state(
        vdp_mod2_rk4,
        base_state,
        p_fixed,
        DT,
        300,
        200,
        x_grid,
        y_grid,
        x_state_index=0,
        y_state_index=1,
        secant_plane=lambda x, y: x[1],
        secant_plane_derivatives=lambda x, y: [0, 1],
        divergence_threshold=1e5,
        n_jobs=-1,
        model=None,
        right_part=vdp_mod2_right_part,
        ode_amplitude_mode="secant",
    )
    Z_nm, _ = grid_of_amplitude_basin_over_initial_state(
        None,
        base_state,
        p_fixed,
        DT,
        50000,
        50000,
        x_grid,
        y_grid,
        x_state_index=0,
        y_state_index=1,
        divergence_threshold=1e5,
        n_jobs=-1,
        model=model,
        ode_amplitude_mode="integrated",
    )
    np.savez(RESULTS_DIR / "e2_amplitude_basin.npz", x_grid=x_grid, y_grid=y_grid, Z_ode=Z_ode, Z_nm=Z_nm)

    lambda_grid = np.linspace(-3.0, 1.0, 20)
    mu_grid = np.linspace(-1.0, 4.0, 20)
    params = [lambda_grid, mu_grid]
    x_init_grid = np.linspace(-3.3, 3.3, 10)
    y_init_grid = np.linspace(-29.0, 29.0, 10)

    P_ode = grid_of_fixed_point_probability_over_params(
        vdp_mod2_rk4,
        base_state,
        params,
        DT,
        4000,
        x_init_grid,
        y_init_grid,
        x_state_index=0,
        y_state_index=1,
        fixed_point_threshold=1e-4,
        divergence_threshold=1e5,
        n_jobs=-1,
        model=None,
    )
    np.savez(RESULTS_DIR / "e2_ode_fixed_point_probability.npz", lambda_grid=lambda_grid, mu_grid=mu_grid, P=P_ode)

    P_nm = grid_of_fixed_point_probability_over_params(
        None,
        base_state,
        params,
        DT,
        4000,
        x_init_grid,
        y_init_grid,
        x_state_index=0,
        y_state_index=1,
        fixed_point_threshold=1e-4,
        divergence_threshold=1e5,
        n_jobs=-1,
        model=model,
    )
    np.savez(RESULTS_DIR / "e2_neuromap_fixed_point_probability.npz", lambda_grid=lambda_grid, mu_grid=mu_grid, P=P_nm)

    u1_axis = np.linspace(-3.3, 3.3, 12)
    u2_axis = np.linspace(-29.0, 29.0, 12)
    dist_Z, log_Z, xg, yg = grid_scan_neuromap_nearest_fixed_point(
        model,
        params,
        (u1_axis, u2_axis),
        residual_tol=1e-7,
        unique_tol=1e-3,
        n_jobs=-1,
    )
    np.savez(RESULTS_DIR / "e2_neuromap_scan_residual_1em7.npz", lambda_grid=xg, mu_grid=yg, dist_Z=dist_Z, log_rho_Z=log_Z, xg=xg, yg=yg)
    print(f"Saved all outputs in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
