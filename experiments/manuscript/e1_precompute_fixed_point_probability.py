#!/usr/bin/env python3
"""
Предвычисление карты P(НТ) по параметрам для ODE и нейроотображения (``grid_of_fixed_point_probability_over_params``).

Запуск из корня репозитория::

    python experiments/manuscript/e1_precompute_fixed_point_probability.py \\
        --checkpoint experiments/manuscript/checkpoints/model.ckpt

Результаты в ``experiments/manuscript/results/`` по умолчанию::

    e1_ode_fixed_point_probability.npz      — alpha_grid, beta_grid, P
    e1_neuromap_fixed_point_probability.npz — P (оси те же, что в ODE-файле)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from neuromaps import NeuroMapManuscript
from systems.vdp_mod1 import vdp_mod1_rk4
from utils import grid_of_fixed_point_probability_over_params

# Как в ``e1_train.py``
DEFAULT_LAMBDA_RANGE = (-3.0, 1.0)
DEFAULT_BETA_RANGE = (0.02, 0.1)
DEFAULT_N_LAMBDA = 45
DEFAULT_N_BETA = 45

DEFAULT_U1_RANGE = (-10.19, 10.18)
DEFAULT_U2_RANGE = (-136.5, 136.5)
DEFAULT_N_IC_U1 = 10
DEFAULT_N_IC_U2 = 10

DEFAULT_DT = 0.01
DEFAULT_N_STEPS = 8000
DEFAULT_FP_THRESHOLD = 1e-10
DEFAULT_DIVERGENCE = 1e5


def _parse_range(s: str) -> tuple[float, float]:
    a, b = s.split(",")
    return float(a.strip()), float(b.strip())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("experiments/manuscript/checkpoints/model.ckpt"),
        help="Чекпоинт NeuroMapManuscript после e1_train.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiments/manuscript/results"),
        help="Каталог для .npz.",
    )
    p.add_argument("--lambda-range", type=_parse_range, default=DEFAULT_LAMBDA_RANGE, help="λ: min,max")
    p.add_argument("--beta-range", type=_parse_range, default=DEFAULT_BETA_RANGE, help="β: min,max")
    p.add_argument("--n-lambda", type=int, default=DEFAULT_N_LAMBDA)
    p.add_argument("--n-beta", type=int, default=DEFAULT_N_BETA)
    p.add_argument("--u1-range", type=_parse_range, default=DEFAULT_U1_RANGE, help="Сетка IC по u₁")
    p.add_argument("--u2-range", type=_parse_range, default=DEFAULT_U2_RANGE, help="Сетка IC по u₂")
    p.add_argument("--n-ic-u1", type=int, default=DEFAULT_N_IC_U1)
    p.add_argument("--n-ic-u2", type=int, default=DEFAULT_N_IC_U2)
    p.add_argument("--dt", type=float, default=DEFAULT_DT)
    p.add_argument("--n-steps", type=int, default=DEFAULT_N_STEPS)
    p.add_argument("--fp-threshold", type=float, default=DEFAULT_FP_THRESHOLD)
    p.add_argument("--divergence", type=float, default=DEFAULT_DIVERGENCE)
    p.add_argument("--jobs", type=int, default=-1, help="Число процессов joblib (-1 = все ядра).")
    args = p.parse_args()

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

    print("ODE: P(НТ) по (λ, β)…")
    P_ode = grid_of_fixed_point_probability_over_params(
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
    np.savez(ode_path, alpha_grid=alpha_grid, beta_grid=beta_grid, P=P_ode)
    print(f"Сохранено: {ode_path}")

    print("Neuromap: P(НТ) по (λ, β)…")
    model = NeuroMapManuscript.load(str(args.checkpoint))
    P_nm = grid_of_fixed_point_probability_over_params(
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
    np.savez(nm_path, P=P_nm)
    print(f"Сохранено: {nm_path}")


if __name__ == "__main__":
    main()
