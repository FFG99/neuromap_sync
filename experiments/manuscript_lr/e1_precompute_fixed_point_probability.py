#!/usr/bin/env python3
"""P(НТ) по (λ, β) для ODE и NeuroMapManuscriptLR — как manuscript_1/e1_precompute_fixed_point_probability.py."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from neuromaps.nm_manuscript_lr import NeuroMapManuscriptLR
from systems.vdp_mod1 import vdp_mod1_rk4
from utils import grid_of_fixed_point_probability_over_params

from experiments.manuscript_lr.e1_config import (
    DEFAULT_VARIANT,
    VARIANT_NAMES,
    model_ckpt_path,
    results_dir,
)

DEFAULT_LAMBDA_RANGE = (-3.0, 1.0)
DEFAULT_BETA_RANGE = (0.02, 0.1)
DEFAULT_N_LAMBDA = 20
DEFAULT_N_BETA = 20
DEFAULT_U1_RANGE = (-10.0, 10.0)
DEFAULT_U2_RANGE = (-10.0, 10.0)
DEFAULT_N_IC_U1 = 10
DEFAULT_N_IC_U2 = 10
DEFAULT_DT = 0.01
DEFAULT_N_STEPS = 4000
DEFAULT_FP_THRESHOLD = 1e-4
DEFAULT_DIVERGENCE = 1e5


def _parse_range(s: str) -> tuple[float, float]:
    a, b = s.split(",")
    return float(a.strip()), float(b.strip())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", choices=VARIANT_NAMES, default=DEFAULT_VARIANT)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--lambda-range", type=_parse_range, default=DEFAULT_LAMBDA_RANGE)
    p.add_argument("--beta-range", type=_parse_range, default=DEFAULT_BETA_RANGE)
    p.add_argument("--n-lambda", type=int, default=DEFAULT_N_LAMBDA)
    p.add_argument("--n-beta", type=int, default=DEFAULT_N_BETA)
    p.add_argument("--u1-range", type=_parse_range, default=DEFAULT_U1_RANGE)
    p.add_argument("--u2-range", type=_parse_range, default=DEFAULT_U2_RANGE)
    p.add_argument("--n-ic-u1", type=int, default=DEFAULT_N_IC_U1)
    p.add_argument("--n-ic-u2", type=int, default=DEFAULT_N_IC_U2)
    p.add_argument("--dt", type=float, default=DEFAULT_DT)
    p.add_argument("--n-steps", type=int, default=DEFAULT_N_STEPS)
    p.add_argument("--fp-threshold", type=float, default=DEFAULT_FP_THRESHOLD)
    p.add_argument("--divergence", type=float, default=DEFAULT_DIVERGENCE)
    p.add_argument("--jobs", type=int, default=-1)
    p.add_argument(
        "--skip-ode",
        action="store_true",
        help="Не считать ODE (если e1_ode_fixed_point_probability.npz уже есть в --out-dir)",
    )
    args = p.parse_args()
    if args.checkpoint is None:
        args.checkpoint = model_ckpt_path(args.variant)
    if args.out_dir is None:
        args.out_dir = results_dir(args.variant)

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Нет чекпоинта: {args.checkpoint}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ode_path = args.out_dir / "e1_ode_fixed_point_probability.npz"
    nm_path = args.out_dir / "e1_neuromap_fixed_point_probability.npz"

    alpha_grid = np.linspace(args.lambda_range[0], args.lambda_range[1], args.n_lambda)
    beta_grid = np.linspace(args.beta_range[0], args.beta_range[1], args.n_beta)
    params = [alpha_grid, beta_grid]
    x_init_grid = np.linspace(args.u1_range[0], args.u1_range[1], args.n_ic_u1)
    y_init_grid = np.linspace(args.u2_range[0], args.u2_range[1], args.n_ic_u2)
    base_state = np.zeros(2, dtype=float)

    if args.skip_ode and ode_path.is_file():
        print(f"ODE: пропуск (--skip-ode), используем {ode_path}")
        ode_data = np.load(ode_path)
        alpha_grid = ode_data["alpha_grid"]
        beta_grid = ode_data["beta_grid"]
    else:
        print("ODE: P(НТ) по (λ, β)…")
        p_ode = grid_of_fixed_point_probability_over_params(
            vdp_mod1_rk4,
            base_state,
            params,
            args.dt,
            args.n_steps,
            x_init_grid,
            y_init_grid,
            x_state_index=0,
            y_state_index=1,
            fixed_point_threshold=args.fp_threshold,
            divergence_threshold=args.divergence,
            n_jobs=args.jobs,
            model=None,
        )
        np.savez(ode_path, alpha_grid=alpha_grid, beta_grid=beta_grid, P=p_ode)
        print(f"Сохранено: {ode_path}")

    print("Neuromap LR: P(НТ) по (λ, β)…")
    model = NeuroMapManuscriptLR.load(str(args.checkpoint), device="cpu")
    p_nm = grid_of_fixed_point_probability_over_params(
        None,
        base_state,
        params,
        args.dt,
        args.n_steps,
        x_init_grid,
        y_init_grid,
        x_state_index=0,
        y_state_index=1,
        fixed_point_threshold=args.fp_threshold,
        divergence_threshold=args.divergence,
        n_jobs=args.jobs,
        model=model,
    )
    np.savez(nm_path, P=p_nm)
    print(f"Сохранено: {nm_path}")


if __name__ == "__main__":
    main()
