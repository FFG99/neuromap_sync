#!/usr/bin/env python3
"""Скан dist / ln ρ по (λ, β) — как manuscript_1/e1_precompute_neuromap_scan_residual.py."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from neuromaps.nm_manuscript_lr import NeuroMapManuscriptLR
from utils import grid_scan_neuromap_nearest_fixed_point

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
        raise ValueError("residual_tol должен быть степенью 10 или задайте --out")
    return out_dir / f"e1_neuromap_scan_residual_1em{-e}.npz"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", choices=VARIANT_NAMES, default=DEFAULT_VARIANT)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--lambda-range", type=_parse_range, default=DEFAULT_LAMBDA_RANGE)
    p.add_argument("--beta-range", type=_parse_range, default=DEFAULT_BETA_RANGE)
    p.add_argument("--n-lambda", type=int, default=DEFAULT_N_LAMBDA)
    p.add_argument("--n-beta", type=int, default=DEFAULT_N_BETA)
    p.add_argument("--u1-start-range", type=_parse_range, default=DEFAULT_U1_START_RANGE)
    p.add_argument("--u2-start-range", type=_parse_range, default=DEFAULT_U2_START_RANGE)
    p.add_argument("--n-u1-start", type=int, default=DEFAULT_N_U1_START)
    p.add_argument("--n-u2-start", type=int, default=DEFAULT_N_U2_START)
    p.add_argument("--residual-tol", type=float, default=DEFAULT_RESIDUAL_TOL)
    p.add_argument("--unique-tol", type=float, default=DEFAULT_UNIQUE_TOL)
    p.add_argument("--jobs", type=int, default=-1)
    args = p.parse_args()
    if args.checkpoint is None:
        args.checkpoint = model_ckpt_path(args.variant)
    if args.out_dir is None:
        args.out_dir = results_dir(args.variant)

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Нет чекпоинта: {args.checkpoint}")
    if args.residual_tol <= 0:
        raise ValueError("residual_tol должен быть положительным.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out if args.out is not None else _default_out_path(args.out_dir, args.residual_tol)

    lam_grid = np.linspace(args.lambda_range[0], args.lambda_range[1], args.n_lambda)
    beta_grid = np.linspace(args.beta_range[0], args.beta_range[1], args.n_beta)
    params = [lam_grid, beta_grid]
    u1_axis = np.linspace(args.u1_start_range[0], args.u1_start_range[1], args.n_u1_start)
    u2_axis = np.linspace(args.u2_start_range[0], args.u2_start_range[1], args.n_u2_start)

    print("Neuromap LR: скан dist / ln ρ…")
    model = NeuroMapManuscriptLR.load(str(args.checkpoint), device="cpu")
    dist_z, log_z, xg, yg = grid_scan_neuromap_nearest_fixed_point(
        model,
        params,
        (u1_axis, u2_axis),
        residual_tol=args.residual_tol,
        unique_tol=args.unique_tol,
        n_jobs=args.jobs,
    )
    np.savez(
        out_path,
        lam_grid=xg,
        beta_grid=yg,
        dist_Z=dist_z,
        log_rho_Z=log_z,
        xg=xg,
        yg=yg,
    )
    print(f"Сохранено: {out_path}")


if __name__ == "__main__":
    main()
