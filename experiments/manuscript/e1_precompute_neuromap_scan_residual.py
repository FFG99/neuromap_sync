#!/usr/bin/env python3
"""
Предвычисление 2D-скана «ближайшая к нулю НТ» для neuromap: расстояние ``‖u*‖`` и
``ln max|μ|`` по сетке (λ, β) — см. ``grid_scan_neuromap_nearest_fixed_point`` в
``utils/nn_map_fixed_points.py``.

Результат в формате, который читает ``e1_analysis.ipynb`` (поля ``lam_grid``,
``beta_grid``, ``dist_Z``, ``log_rho_Z``, ``xg``, ``yg``).

Запуск из корня репозитория::

    python experiments/manuscript/e1_precompute_neuromap_scan_residual.py \\
        --checkpoint experiments/manuscript/checkpoints/model.ckpt \\
        --residual-tol 1e-7

По умолчанию имя файла ``e1_neuromap_scan_residual_1em7.npz`` для ``1e-7``;
для других порогов, если имя не получается однозначно из степени 10, задайте ``--out``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from neuromaps import NeuroMapManuscript
from utils import grid_scan_neuromap_nearest_fixed_point

# Как в e1_precompute_fixed_point_probability / ноутбук: сетка по параметрам
DEFAULT_LAMBDA_RANGE = (-3.0, 1.0)
DEFAULT_BETA_RANGE = (0.02, 0.1)
DEFAULT_N_LAMBDA = 20
DEFAULT_N_BETA = 20

# Как в ячейке e1_analysis с collect_fixed_points_grid_starts (фазовый портрет)
DEFAULT_U1_START_RANGE = (-15.0, 15.0)
DEFAULT_U2_START_RANGE = (-35.0, 35.0)
DEFAULT_N_U1_START = 12
DEFAULT_N_U2_START = 12

DEFAULT_RESIDUAL_TOL = 1e-7
DEFAULT_UNIQUE_TOL = 1e-3


def _parse_range(s: str) -> tuple[float, float]:
    a, b = s.split(",")
    return float(a.strip()), float(b.strip())


def _default_out_path(out_dir: Path, residual_tol: float) -> Path:
    e = int(round(np.log10(residual_tol)))
    if not np.isclose(residual_tol, 10.0**e, rtol=0.0, atol=1e-18):
        raise ValueError(
            "Для автоматического имени ``residual_tol`` должен быть точной степенью 10 "
            "(например 1e-7). Иначе укажите ``--out`` явно."
        )
    tag = f"1em{-e}"
    return out_dir / f"e1_neuromap_scan_residual_{tag}.npz"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("experiments/manuscript/checkpoints/model.ckpt"),
        help="Чекпоинт NeuroMapManuscript.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("experiments/manuscript/results"),
        help="Каталог для .npz (если не задан --out).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Полный путь к .npz; по умолчанию out-dir / e1_neuromap_scan_residual_<tag>.npz",
    )
    p.add_argument("--lambda-range", type=_parse_range, default=DEFAULT_LAMBDA_RANGE, help="λ: min,max")
    p.add_argument("--beta-range", type=_parse_range, default=DEFAULT_BETA_RANGE, help="β: min,max")
    p.add_argument("--n-lambda", type=int, default=DEFAULT_N_LAMBDA)
    p.add_argument("--n-beta", type=int, default=DEFAULT_N_BETA)
    p.add_argument("--u1-start-range", type=_parse_range, default=DEFAULT_U1_START_RANGE, help="Ось стартов u₁")
    p.add_argument("--u2-start-range", type=_parse_range, default=DEFAULT_U2_START_RANGE, help="Ось стартов u₂")
    p.add_argument("--n-u1-start", type=int, default=DEFAULT_N_U1_START)
    p.add_argument("--n-u2-start", type=int, default=DEFAULT_N_U2_START)
    p.add_argument("--residual-tol", type=float, default=DEFAULT_RESIDUAL_TOL)
    p.add_argument("--unique-tol", type=float, default=DEFAULT_UNIQUE_TOL)
    p.add_argument("--jobs", type=int, default=-1, help="Число процессов joblib (-1 = все ядра).")
    args = p.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Нет чекпоинта: {args.checkpoint}")
    if args.residual_tol <= 0:
        raise ValueError("residual_tol должен быть положительным.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out if args.out is not None else _default_out_path(args.out_dir, args.residual_tol)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lam_grid = np.linspace(args.lambda_range[0], args.lambda_range[1], args.n_lambda)
    beta_grid = np.linspace(args.beta_range[0], args.beta_range[1], args.n_beta)
    params = [lam_grid, beta_grid]

    u1_axis = np.linspace(args.u1_start_range[0], args.u1_start_range[1], args.n_u1_start)
    u2_axis = np.linspace(args.u2_start_range[0], args.u2_start_range[1], args.n_u2_start)
    u_start_axes = (u1_axis, u2_axis)

    print("Neuromap: скан dist / ln ρ по (λ, β)…")
    model = NeuroMapManuscript.load(str(args.checkpoint), device="cpu")

    dist_Z, log_Z, xg, yg = grid_scan_neuromap_nearest_fixed_point(
        model,
        params,
        u_start_axes,
        residual_tol=args.residual_tol,
        unique_tol=args.unique_tol,
        n_jobs=args.jobs,
    )

    np.savez(
        out_path,
        lam_grid=xg,
        beta_grid=yg,
        dist_Z=dist_Z,
        log_rho_Z=log_Z,
        xg=xg,
        yg=yg,
    )
    print(f"Сохранено: {out_path}")


if __name__ == "__main__":
    main()
