#!/usr/bin/env python3
"""
Предвычисление басейнов притяжения (сетка амплитуд по начальным условиям) для ODE и нейроотображения.

Дефолты как в ``e1_analysis.ipynb`` до выноса скриптов (commit 4ee0db8273364a58e928c76c62ee7686b5519c3f):
  - ODE: ``ode_amplitude_mode="secant"``, ``n_transient=300``, ``n_attractor=200``
  - Neuromap: ``ode_amplitude_mode="integrated"``, ``n_transient=50000``, ``n_attractor=50000``

Запуск из корня репозитория::

    python experiments/manuscript/e1_precompute_amplitude_basins.py \\
        --checkpoint experiments/manuscript/checkpoints/model.ckpt

Результат (по умолчанию ``experiments/manuscript/results/e1_amplitude_basin.npz``)::

    x_grid, y_grid, Z_ode, Z_nm
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from neuromaps import NeuroMapManuscript
from systems.vdp_mod1 import vdp_mod1_rk4, vdp_mod1_right_part
from utils import grid_of_amplitude_basin_over_initial_state

# Как в старом ноутбуке: linspace(-10, 10, 100), linspace(-135, 135, 100)
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


def precompute_amplitude_basin(
    *,
    checkpoint: Path,
    out_path: Path,
    u1_range: tuple[float, float],
    u2_range: tuple[float, float],
    n_u1: int,
    n_u2: int,
    lam: float,
    beta: float,
    dt: float,
    ode_n_transient: int,
    ode_n_attractor: int,
    nm_n_transient: int,
    nm_n_attractor: int,
    divergence_threshold: float,
    n_jobs: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x_grid = np.linspace(u1_range[0], u1_range[1], n_u1)
    y_grid = np.linspace(u2_range[0], u2_range[1], n_u2)
    base_state = np.zeros(2, dtype=float)
    p_fixed = [lam, beta]

    print("ODE: сетка амплитуд по (u₁, u₂), режим secant…")
    Z_ode, _mask_ode = grid_of_amplitude_basin_over_initial_state(
        vdp_mod1_rk4,
        base_state,
        p_fixed,
        dt,
        ode_n_transient,
        ode_n_attractor,
        x_grid,
        y_grid,
        x_state_index=0,
        y_state_index=1,
        secant_plane=lambda x, y: x[1],
        secant_plane_derivatives=lambda x, y: [0, 1],
        divergence_threshold=divergence_threshold,
        n_jobs=n_jobs,
        model=None,
        right_part=vdp_mod1_right_part,
        ode_amplitude_mode="secant",
    )

    print("Neuromap: та же сетка, integrated…")
    model = NeuroMapManuscript.load(str(checkpoint), device="cpu")
    Z_nm, _mask_nm = grid_of_amplitude_basin_over_initial_state(
        None,
        base_state,
        p_fixed,
        dt,
        nm_n_transient,
        nm_n_attractor,
        x_grid,
        y_grid,
        x_state_index=0,
        y_state_index=1,
        divergence_threshold=divergence_threshold,
        n_jobs=n_jobs,
        model=model,
        ode_amplitude_mode="integrated",
    )

    np.savez(
        out_path,
        x_grid=x_grid,
        y_grid=y_grid,
        Z_ode=Z_ode,
        Z_nm=Z_nm,
    )
    print(f"Сохранено: {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("experiments/manuscript/checkpoints/model.ckpt"),
        help="Чекпоинт NeuroMapManuscript после e1_train.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("experiments/manuscript/results/e1_amplitude_basin.npz"),
        help="Путь к выходному .npz.",
    )
    p.add_argument("--u1-range", type=_parse_range, default=DEFAULT_U1_RANGE, help="u₁: min,max")
    p.add_argument("--u2-range", type=_parse_range, default=DEFAULT_U2_RANGE, help="u₂: min,max")
    p.add_argument("--n-u1", type=int, default=DEFAULT_N_U1)
    p.add_argument("--n-u2", type=int, default=DEFAULT_N_U2)
    p.add_argument("--lambda", dest="lam", type=float, default=DEFAULT_LAMBDA, help="Параметр λ")
    p.add_argument("--beta", type=float, default=DEFAULT_BETA, help="Параметр β")
    p.add_argument("--dt", type=float, default=DEFAULT_DT)
    p.add_argument("--ode-n-transient", type=int, default=DEFAULT_ODE_N_TRANSIENT)
    p.add_argument("--ode-n-attractor", type=int, default=DEFAULT_ODE_N_ATTRACTOR)
    p.add_argument("--nm-n-transient", type=int, default=DEFAULT_NM_N_TRANSIENT)
    p.add_argument("--nm-n-attractor", type=int, default=DEFAULT_NM_N_ATTRACTOR)
    p.add_argument("--divergence", type=float, default=DEFAULT_DIVERGENCE)
    p.add_argument("--jobs", type=int, default=-1, help="Число процессов joblib (-1 = все ядра).")
    args = p.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Нет чекпоинта: {args.checkpoint}")

    precompute_amplitude_basin(
        checkpoint=args.checkpoint,
        out_path=args.out,
        u1_range=args.u1_range,
        u2_range=args.u2_range,
        n_u1=args.n_u1,
        n_u2=args.n_u2,
        lam=args.lam,
        beta=args.beta,
        dt=args.dt,
        ode_n_transient=args.ode_n_transient,
        ode_n_attractor=args.ode_n_attractor,
        nm_n_transient=args.nm_n_transient,
        nm_n_attractor=args.nm_n_attractor,
        divergence_threshold=args.divergence,
        n_jobs=args.jobs,
    )


if __name__ == "__main__":
    main()
