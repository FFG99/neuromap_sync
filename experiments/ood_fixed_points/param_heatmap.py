"""
2D heatmap of rollout error vs (lambda, beta) for trained NeuroMap checkpoints.

Run from repository root:
  python experiments/ood_fixed_points/param_heatmap.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from tqdm import tqdm

from systems.vdp_mod1 import vdp_mod1_rk4
from neuromaps import (
    NeuroMapOriginal,
    NeuroMapTargetNormalized,
    NeuroMapManuscript,
)

DT = 0.01
ROLLOUT_DIVERGENCE_THRESHOLD = 2e3
K_FOCUS = 30

ROOT_DIR = Path("experiments/ood_fixed_points")
CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"
RESULTS_DIR = ROOT_DIR / "results" / "param_grid"

# Train rectangle (same as train.py) — drawn on figures
PARAMETERS_RANGES_TRAIN = [(-2.0, 0.4), (0.03, 0.08)]


def true_rollout(u0: np.ndarray, p: np.ndarray, n_steps: int, dt: float) -> np.ndarray:
    u = np.array(u0, dtype=np.float64)
    out = [u.copy()]
    for _ in range(n_steps):
        u = vdp_mod1_rk4(u, p, dt)
        if not np.all(np.isfinite(u)) or np.linalg.norm(u) > ROLLOUT_DIVERGENCE_THRESHOLD:
            break
        out.append(u.copy())
    return np.array(out)


def model_rollout(model, u0: np.ndarray, p: np.ndarray, n_steps: int) -> tuple[np.ndarray, bool]:
    u = np.array(u0, dtype=np.float64)
    out = [u.copy()]
    diverged = False
    for _ in range(n_steps):
        X = np.concatenate([u, p], axis=0)[None, :]
        d = model.predict(X)[0]
        u = u + d
        if not np.all(np.isfinite(u)) or np.linalg.norm(u) > ROLLOUT_DIVERGENCE_THRESHOLD:
            diverged = True
            break
        out.append(u.copy())
    return np.array(out), diverged


def traj_rollout_nrmse_and_k30(
    model,
    u0: np.ndarray,
    p: np.ndarray,
    n_steps: int,
    dt: float,
    k_focus: int,
) -> tuple[float, float]:
    true_traj = true_rollout(u0, p, n_steps, dt)
    pred_traj, _ = model_rollout(model, u0, p, n_steps)

    rollout_sqerr = 0.0
    rollout_true_sq = 0.0
    m = min(len(pred_traj), len(true_traj))
    if m > 0:
        diff = pred_traj[:m] - true_traj[:m]
        valid = np.isfinite(diff).all(axis=1)
        if np.any(valid):
            diff = diff[valid]
            tt = true_traj[:m][valid]
            rollout_sqerr = float(np.sum(diff * diff))
            rollout_true_sq = float(np.sum(tt * tt))

    rollout_nrmse = (
        float(np.sqrt(rollout_sqerr / rollout_true_sq)) if rollout_true_sq > 0 else float("nan")
    )

    m_state = min(len(pred_traj), len(true_traj))
    k30 = float("nan")
    if m_state > k_focus:
        pred_k = pred_traj[k_focus:m_state]
        true_k = true_traj[k_focus:m_state]
        valid_k = np.isfinite(pred_k).all(axis=1) & np.isfinite(true_k).all(axis=1)
        if np.any(valid_k):
            k30 = float(np.mean(np.abs(pred_k[valid_k] - true_k[valid_k])))

    return rollout_nrmse, k30


def load_model(model_name: str):
    cls_map = {
        "original": NeuroMapOriginal,
        "target_normalized": NeuroMapTargetNormalized,
        "manuscript": NeuroMapManuscript,
    }
    ckpt = CHECKPOINTS_DIR / model_name / "model.ckpt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    return cls_map[model_name].load(str(ckpt))


def build_initial_conditions(rng: np.random.Generator, n_ic: int) -> list[np.ndarray]:
    """Small set of reproducible starts: fixed anchors + random fill."""
    anchors = [
        np.array([0.0, 0.0], dtype=np.float64),
        np.array([1.0, 0.0], dtype=np.float64),
        np.array([-1.0, 5.0], dtype=np.float64),
        np.array([2.0, -10.0], dtype=np.float64),
    ]
    out = anchors[: min(len(anchors), n_ic)]
    while len(out) < n_ic:
        out.append(rng.uniform(-6.0, 6.0, size=2).astype(np.float64))
    return out[:n_ic]


def run_grid(
    model,
    lambda_axis: np.ndarray,
    beta_axis: np.ndarray,
    ic_list: list[np.ndarray],
    n_steps: int,
    dt: float,
    k_focus: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (rollout_nrmse_grid, k30_mae_grid) shape (len(beta), len(lambda))."""
    n_beta, n_lam = len(beta_axis), len(lambda_axis)
    g_nrmse = np.full((n_beta, n_lam), np.nan, dtype=np.float64)
    g_k30 = np.full((n_beta, n_lam), np.nan, dtype=np.float64)

    for bi, beta in enumerate(tqdm(beta_axis, desc="beta rows", leave=False)):
        for li, lam in enumerate(lambda_axis):
            p = np.array([lam, beta], dtype=np.float64)
            nrmse_vals = []
            k30_vals = []
            for u0 in ic_list:
                nrmse, k30 = traj_rollout_nrmse_and_k30(model, u0, p, n_steps, dt, k_focus)
                nrmse_vals.append(nrmse)
                k30_vals.append(k30)
            g_nrmse[bi, li] = np.nanmean(nrmse_vals)
            g_k30[bi, li] = np.nanmean(k30_vals)

    return g_nrmse, g_k30


def plot_heatmap(
    values: np.ndarray,
    lambda_axis: np.ndarray,
    beta_axis: np.ndarray,
    title: str,
    cbar_label: str,
    out_path: Path,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    extent = (
        float(lambda_axis[0]),
        float(lambda_axis[-1]),
        float(beta_axis[0]),
        float(beta_axis[-1]),
    )
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    im = ax.imshow(
        values,
        origin="lower",
        aspect="auto",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$\beta$")
    ax.set_title(title)
    la0, la1 = PARAMETERS_RANGES_TRAIN[0]
    b0, b1 = PARAMETERS_RANGES_TRAIN[1]
    rect_w = la1 - la0
    rect_h = b1 - b0
    ax.add_patch(
        Rectangle(
            (la0, b0),
            rect_w,
            rect_h,
            linewidth=1.5,
            edgecolor="white",
            facecolor="none",
            linestyle="--",
        )
    )
    plt.colorbar(im, ax=ax, label=cbar_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parameter heatmap for OOD experiment.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smaller grid and fewer steps for a fast sanity check.",
    )
    args = parser.parse_args()

    if args.quick:
        n_lam, n_beta = 15, 15
        n_steps = 100
        n_ic = 4
        seed = 9001
    else:
        n_lam, n_beta = 31, 31
        n_steps = 200
        n_ic = 8
        seed = 42

    lambda_axis = np.linspace(-2.3, 1.05, n_lam)
    beta_axis = np.linspace(0.02, 0.115, n_beta)

    rng = np.random.default_rng(seed)
    ic_list = build_initial_conditions(rng, n_ic)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    meta = {
        "dt": DT,
        "n_steps": n_steps,
        "k_focus": K_FOCUS,
        "n_ic": n_ic,
        "seed": seed,
        "lambda_axis": lambda_axis.tolist(),
        "beta_axis": beta_axis.tolist(),
        "train_box": PARAMETERS_RANGES_TRAIN,
    }
    with open(RESULTS_DIR / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    rows_csv = []
    for model_name in tqdm(
        ["original", "target_normalized", "manuscript"],
        desc="Models",
        unit="model",
    ):
        model = load_model(model_name)
        g_nrmse, g_k30 = run_grid(
            model,
            lambda_axis,
            beta_axis,
            ic_list,
            n_steps=n_steps,
            dt=DT,
            k_focus=K_FOCUS,
        )
        np.savez_compressed(
            RESULTS_DIR / f"{model_name}_grids.npz",
            rollout_nrmse=g_nrmse,
            k30_mae=g_k30,
            lambda_axis=lambda_axis,
            beta_axis=beta_axis,
        )

        for bi, beta in enumerate(beta_axis):
            for li, lam in enumerate(lambda_axis):
                rows_csv.append(
                    {
                        "model": model_name,
                        "lambda": float(lam),
                        "beta": float(beta),
                        "rollout_nrmse": float(g_nrmse[bi, li]),
                        "k30_mae": float(g_k30[bi, li]),
                    }
                )

        # Robust color limits (clip outliers for readability)
        v1, v99 = np.nanpercentile(g_nrmse, [1, 99])
        plot_heatmap(
            g_nrmse,
            lambda_axis,
            beta_axis,
            title=f"{model_name}: rollout NRMSE (train box dashed)",
            cbar_label="NRMSE",
            out_path=RESULTS_DIR / f"{model_name}_rollout_nrmse.png",
            vmin=float(v1),
            vmax=float(v99),
        )
        v1k, v99k = np.nanpercentile(g_k30, [1, 99])
        plot_heatmap(
            g_k30,
            lambda_axis,
            beta_axis,
            title=f"{model_name}: |u_pred − u_true| at k={K_FOCUS} (mean over ICs)",
            cbar_label="MAE",
            out_path=RESULTS_DIR / f"{model_name}_k30_mae.png",
            vmin=float(v1k),
            vmax=float(v99k),
        )

    csv_path = RESULTS_DIR / "grid_long.csv"
    if rows_csv:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_csv[0].keys()))
            w.writeheader()
            w.writerows(rows_csv)

    print(f"Saved grids, CSV, PNG under {RESULTS_DIR}")


if __name__ == "__main__":
    main()
