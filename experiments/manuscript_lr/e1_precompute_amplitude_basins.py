#!/usr/bin/env python3
"""Басейны амплитуд (ODE secant + Neuromap integrated) — как manuscript_1/e1_precompute_amplitude_basins.py."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from neuromaps.nm_manuscript_lr import NeuroMapManuscriptLR
from systems.vdp_mod1 import vdp_mod1_rk4, vdp_mod1_right_part
from utils import grid_of_amplitude_basin_over_initial_state

from experiments.manuscript_lr.e1_config import (
    DEFAULT_VARIANT,
    VARIANT_NAMES,
    model_ckpt_path,
    results_dir,
)

DEFAULT_U1_RANGE = (-10.0, 10.0)
DEFAULT_U2_RANGE = (-135.0, 135.0)
DEFAULT_N_U1 = 100
DEFAULT_N_U2 = 100
DEFAULT_LAMBDA = -1.0
DEFAULT_BETA = 0.07
DEFAULT_DT = 0.01
DEFAULT_ODE_N_TRANSIENT = 300
DEFAULT_ODE_N_ATTRACTOR = 200
DEFAULT_NM_N_TRANSIENT = 50000
DEFAULT_NM_N_ATTRACTOR = 50000
DEFAULT_DIVERGENCE = 1e5


def _parse_range(s: str) -> tuple[float, float]:
    a, b = s.split(",")
    return float(a.strip()), float(b.strip())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", choices=VARIANT_NAMES, default=DEFAULT_VARIANT)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--u1-range", type=_parse_range, default=DEFAULT_U1_RANGE)
    p.add_argument("--u2-range", type=_parse_range, default=DEFAULT_U2_RANGE)
    p.add_argument("--n-u1", type=int, default=DEFAULT_N_U1)
    p.add_argument("--n-u2", type=int, default=DEFAULT_N_U2)
    p.add_argument("--lambda", dest="lam", type=float, default=DEFAULT_LAMBDA)
    p.add_argument("--beta", type=float, default=DEFAULT_BETA)
    p.add_argument("--dt", type=float, default=DEFAULT_DT)
    p.add_argument("--ode-n-transient", type=int, default=DEFAULT_ODE_N_TRANSIENT)
    p.add_argument("--ode-n-attractor", type=int, default=DEFAULT_ODE_N_ATTRACTOR)
    p.add_argument("--nm-n-transient", type=int, default=DEFAULT_NM_N_TRANSIENT)
    p.add_argument("--nm-n-attractor", type=int, default=DEFAULT_NM_N_ATTRACTOR)
    p.add_argument("--divergence", type=float, default=DEFAULT_DIVERGENCE)
    p.add_argument("--jobs", type=int, default=-1)
    args = p.parse_args()
    if args.checkpoint is None:
        args.checkpoint = model_ckpt_path(args.variant)
    if args.out is None:
        args.out = results_dir(args.variant) / "e1_amplitude_basin.npz"

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Нет чекпоинта: {args.checkpoint}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    x_grid = np.linspace(args.u1_range[0], args.u1_range[1], args.n_u1)
    y_grid = np.linspace(args.u2_range[0], args.u2_range[1], args.n_u2)
    base_state = np.zeros(2, dtype=float)
    p_fixed = [args.lam, args.beta]

    print("ODE: сетка амплитуд, secant…")
    z_ode, _ = grid_of_amplitude_basin_over_initial_state(
        vdp_mod1_rk4,
        base_state,
        p_fixed,
        args.dt,
        args.ode_n_transient,
        args.ode_n_attractor,
        x_grid,
        y_grid,
        x_state_index=0,
        y_state_index=1,
        secant_plane=lambda x, y: x[1],
        secant_plane_derivatives=lambda x, y: [0, 1],
        divergence_threshold=args.divergence,
        n_jobs=args.jobs,
        model=None,
        right_part=vdp_mod1_right_part,
        ode_amplitude_mode="secant",
    )

    print("Neuromap LR: integrated…")
    model = NeuroMapManuscriptLR.load(str(args.checkpoint), device="cpu")
    z_nm, _ = grid_of_amplitude_basin_over_initial_state(
        None,
        base_state,
        p_fixed,
        args.dt,
        args.nm_n_transient,
        args.nm_n_attractor,
        x_grid,
        y_grid,
        x_state_index=0,
        y_state_index=1,
        divergence_threshold=args.divergence,
        n_jobs=args.jobs,
        model=model,
        ode_amplitude_mode="integrated",
    )

    np.savez(args.out, x_grid=x_grid, y_grid=y_grid, Z_ode=z_ode, Z_nm=z_nm)
    print(f"Сохранено: {args.out}")


if __name__ == "__main__":
    main()
