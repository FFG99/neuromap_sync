import argparse
from pathlib import Path

import numpy as np

from neuromaps import NeuroMapManuscript
from utils import grid_of_fixed_point_probability_over_params
from utils.nn_map_fixed_points import grid_scan_neuromap_nearest_fixed_point


ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = ROOT / "checkpoints" / "model.ckpt"
RESULTS_DIR = ROOT / "results"


def _ensure_results_dir() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _load_model() -> NeuroMapManuscript:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {CHECKPOINT_PATH}")
    return NeuroMapManuscript.load(str(CHECKPOINT_PATH))


def compute_ode_fixed_point_probability(model: NeuroMapManuscript) -> Path:
    # Use denser grids for publication-quality heatmaps (not quick checks).
    alpha_grid = np.linspace(-3.0, 1.0, 40)
    beta_grid = np.linspace(0.02, 0.1, 40)

    P = grid_of_fixed_point_probability_over_params(
        evolution_operator=None,
        model=model,
        state=[0.0, 0.0],
        params=[alpha_grid, beta_grid],
        dt=0.01,
        n_steps=8000,
        x_init_grid=np.linspace(-10, 10, 20),
        y_init_grid=np.linspace(-10, 10, 20),
        x_state_index=0,
        y_state_index=1,
        fixed_point_threshold=1e-4,
        divergence_threshold=1e5,
        n_jobs=-1,
    )

    out_path = RESULTS_DIR / "e1_ode_fixed_point_probability.npz"
    np.savez_compressed(
        out_path,
        alpha_grid=alpha_grid,
        beta_grid=beta_grid,
        P=P,
    )
    return out_path


def compute_neuromap_scan(model: NeuroMapManuscript, residual_tol: float) -> Path:
    # Denser start grid improves fixed-point search robustness in parameter scan.
    u0_u1 = np.linspace(-15, 15, 20)
    u0_u2 = np.linspace(-35, 35, 20)
    lam_grid = np.linspace(-3.0, 1.0, 60)
    beta_grid = np.linspace(0.02, 0.1, 60)

    dist_Z, log_rho_Z, xg, yg = grid_scan_neuromap_nearest_fixed_point(
        model,
        [lam_grid, beta_grid],
        (u0_u1, u0_u2),
        n_jobs=-1,
        residual_tol=residual_tol,
        unique_tol=1e-3,
    )

    tol_tag = str(residual_tol).replace("-", "m").replace(".", "p")
    out_path = RESULTS_DIR / f"e1_neuromap_scan_residual_{tol_tag}.npz"
    np.savez_compressed(
        out_path,
        residual_tol=np.array([residual_tol], dtype=float),
        lam_grid=lam_grid,
        beta_grid=beta_grid,
        dist_Z=dist_Z,
        log_rho_Z=log_rho_Z,
        xg=xg,
        yg=yg,
        u0_u1=u0_u1,
        u0_u2=u0_u2,
    )
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute heavy E1 grid calculations for analysis notebook."
    )
    parser.add_argument(
        "--target",
        choices=["all", "ode", "neuromap"],
        default="all",
        help="Which data to precompute.",
    )
    parser.add_argument(
        "--residual-tols",
        default="1e-7,1e-10,1e-8",
        help="Comma-separated residual tolerances for neuromap scan.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _ensure_results_dir()
    model = _load_model()

    if args.target in ("all", "ode"):
        out = compute_ode_fixed_point_probability(model)
        print(f"[OK] saved ODE fixed-point probability grid: {out}")

    if args.target in ("all", "neuromap"):
        residual_tols = [float(x.strip()) for x in args.residual_tols.split(",") if x.strip()]
        for tol in residual_tols:
            out = compute_neuromap_scan(model, tol)
            print(f"[OK] saved neuromap grid for residual_tol={tol:g}: {out}")


if __name__ == "__main__":
    main()
