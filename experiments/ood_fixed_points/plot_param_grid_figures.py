"""
Build publication-style figures from param_grid/grid_long.csv (and optional .npz).

Run from repo root after param_heatmap.py:
  python experiments/ood_fixed_points/plot_param_grid_figures.py

Outputs:
  - results/param_grid/figure_combined_rollout_nrmse.png
  - results/param_grid/figure_combined_k30_mae.png
  - results/param_grid/figure_slice_beta_mid_rollout_nrmse.png
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import numpy as np

ROOT = Path("experiments/ood_fixed_points/results/param_grid")
META_PATH = ROOT / "meta.json"
CSV_PATH = ROOT / "grid_long.csv"

MODELS = ["original", "target_normalized", "manuscript"]
MODEL_LABELS = {
    "original": "Original",
    "target_normalized": "Target-normalized",
    "manuscript": "Manuscript",
}


def load_long_csv() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[float], list[float]]:
    rows = list(csv.DictReader(CSV_PATH.open(encoding="utf-8")))
    lams = sorted({float(r["lambda"]) for r in rows})
    betas = sorted({float(r["beta"]) for r in rows})
    li = {v: i for i, v in enumerate(lams)}
    bi = {v: i for i, v in enumerate(betas)}

    nrmse = {m: np.full((len(betas), len(lams)), np.nan) for m in MODELS}
    k30 = {m: np.full((len(betas), len(lams)), np.nan) for m in MODELS}

    for r in rows:
        m = r["model"]
        if m not in MODELS:
            continue
        i = li[float(r["lambda"])]
        j = bi[float(r["beta"])]
        nrmse[m][j, i] = float(r["rollout_nrmse"])
        k30[m][j, i] = float(r["k30_mae"])

    return nrmse, k30, lams, betas


def train_box_from_meta(meta: dict) -> tuple[tuple[float, float], tuple[float, float]]:
    tb = meta["train_box"]
    return (tb[0][0], tb[0][1]), (tb[1][0], tb[1][1])


def plot_combined_panels(
    grids: dict[str, np.ndarray],
    lams: list[float],
    betas: list[float],
    train_l: tuple[float, float],
    train_b: tuple[float, float],
    title_prefix: str,
    cbar_label: str,
    out_name: str,
    percentile_clip: tuple[float, float] = (2, 98),
) -> None:
    arrs = [grids[m] for m in MODELS]
    stacked = np.concatenate([a.ravel() for a in arrs])
    stacked = stacked[np.isfinite(stacked)]
    vmin = float(np.percentile(stacked, percentile_clip[0]))
    vmax = float(np.percentile(stacked, percentile_clip[1]))
    if vmax <= vmin:
        vmax = vmin + 1e-6

    extent = (lams[0], lams[-1], betas[0], betas[-1])
    fig = plt.figure(figsize=(15.0, 5.0))
    gs = gridspec.GridSpec(
        1,
        4,
        figure=fig,
        width_ratios=[1.0, 1.0, 1.0, 0.055],
        wspace=0.32,
        left=0.07,
        right=0.98,
        bottom=0.12,
        top=0.88,
    )
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1], sharey=ax0)
    ax2 = fig.add_subplot(gs[0, 2], sharey=ax0)
    cax = fig.add_subplot(gs[0, 3])
    axes = (ax0, ax1, ax2)

    ims = []
    for ax, m in zip(axes, MODELS):
        im = ax.imshow(
            grids[m],
            origin="lower",
            aspect="auto",
            extent=extent,
            vmin=vmin,
            vmax=vmax,
        )
        ims.append(im)
        ax.set_xlabel(r"$\lambda$")
        ax.set_title(MODEL_LABELS[m])
        la0, la1 = train_l
        b0, b1 = train_b
        ax.add_patch(
            Rectangle(
                (la0, b0),
                la1 - la0,
                b1 - b0,
                linewidth=1.8,
                edgecolor="white",
                facecolor="none",
                linestyle="--",
            )
        )
    ax0.set_ylabel(r"$\beta$")
    plt.setp(ax1.get_yticklabels(), visible=False)
    plt.setp(ax2.get_yticklabels(), visible=False)

    fig.suptitle(title_prefix + " (shared color scale; dashed = train region)", fontsize=12)
    cbar = fig.colorbar(ims[0], cax=cax)
    cbar.set_label(cbar_label)
    cax.yaxis.set_ticks_position("right")
    cax.yaxis.set_label_position("right")
    out = ROOT / out_name
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def plot_lambda_slice(
    grids: dict[str, np.ndarray],
    lams: list[float],
    betas: list[float],
    train_l: tuple[float, float],
    beta_pick: float,
    metric_name: str,
    out_name: str,
) -> None:
    """One line per model: metric vs lambda at beta closest to beta_pick."""
    j = int(np.argmin(np.abs(np.array(betas) - beta_pick)))
    beta_val = betas[j]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for m in MODELS:
        ax.plot(lams, grids[m][j, :], label=MODEL_LABELS[m], linewidth=1.8)
    ax.axvspan(train_l[0], train_l[1], alpha=0.12, color="gray", label="Train $\\lambda$ range")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(metric_name)
    ax.set_title(f"{metric_name} vs $\\lambda$ at $\\beta \\approx {beta_val:.4f}$")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = ROOT / out_name
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing {CSV_PATH}; run param_heatmap.py first.")

    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    train_l, train_b = train_box_from_meta(meta)
    beta_mid = 0.5 * (train_b[0] + train_b[1])

    nrmse, k30, lams, betas = load_long_csv()

    plot_combined_panels(
        nrmse,
        lams,
        betas,
        train_l,
        train_b,
        title_prefix="Rollout NRMSE",
        cbar_label="NRMSE",
        out_name="figure_combined_rollout_nrmse.png",
    )
    plot_combined_panels(
        k30,
        lams,
        betas,
        train_l,
        train_b,
        title_prefix=r"$k=30$ state MAE",
        cbar_label="MAE",
        out_name="figure_combined_k30_mae.png",
    )

    plot_lambda_slice(
        nrmse,
        lams,
        betas,
        train_l,
        beta_mid,
        metric_name="Rollout NRMSE",
        out_name="figure_slice_beta_mid_rollout_nrmse.png",
    )
    plot_lambda_slice(
        k30,
        lams,
        betas,
        train_l,
        beta_mid,
        metric_name=r"$k=30$ MAE",
        out_name="figure_slice_beta_mid_k30_mae.png",
    )


if __name__ == "__main__":
    main()
