from pathlib import Path
import csv
import json

import numpy as np
from tqdm import tqdm

from systems.vdp_mod1 import vdp_mod1_rk4
from utils import collect_fixed_points_trajectory_grid
from neuromaps import (
    NeuroMapOriginal,
    NeuroMapTargetNormalized,
    NeuroMapManuscript,
)


SEED = 321
DT = 0.01

VARIABLES_RANGES = [(-8.0, 8.0), (-80.0, 80.0)]
PARAMETERS_RANGES_ID = [(-2.0, 0.4), (0.03, 0.08)]
PARAMETERS_RANGES_OOD = [(0.4, 1.0), (0.08, 0.11)]

N_TRAJ_EVAL = 240
N_STEPS_EVAL = 200
ROLLOUT_DIVERGENCE_THRESHOLD = 2e3
K_STEPS = (1, 5, 10, 30)

FP_PARAM_SAMPLES = 36
FP_AXIS = [
    np.linspace(-3.0, 3.0, 9),
    np.linspace(-30.0, 30.0, 9),
]
FP_MAX_ITER = 12_000
FP_D_TOL = 2e-3
FP_UNIQUE_TOL = 1e-3

ROOT_DIR = Path("experiments/ood_fixed_points")
CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"
RESULTS_DIR = ROOT_DIR / "results"


def sample_initial_conditions(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.uniform(*zip(*VARIABLES_RANGES), size=(n, 2))


def sample_params(rng: np.random.Generator, n: int, ranges) -> np.ndarray:
    return rng.uniform(*zip(*ranges), size=(n, 2))


def true_rollout(u0: np.ndarray, p: np.ndarray, n_steps: int) -> np.ndarray:
    u = np.array(u0, dtype=np.float64)
    out = [u.copy()]
    for _ in range(n_steps):
        u = vdp_mod1_rk4(u, p, DT)
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


def evaluate_split(model, params: np.ndarray, initials: np.ndarray) -> dict:
    one_step_abs = []
    k_step_abs = {k: [] for k in K_STEPS}
    rollout_sqerr = []
    rollout_true_sq = []
    rollout_points = 0
    pred_diverged_count = 0
    true_invalid_count = 0
    pred_invalid_count = 0

    for p, u0 in tqdm(
        zip(params, initials),
        total=len(params),
        desc="Rollout eval",
        unit="traj",
        leave=False,
    ):
        true_traj = true_rollout(u0, p, N_STEPS_EVAL)
        pred_traj, diverged = model_rollout(model, u0, p, N_STEPS_EVAL)
        if diverged:
            pred_diverged_count += 1

        if len(true_traj) < N_STEPS_EVAL + 1:
            true_invalid_count += 1
        if len(pred_traj) < N_STEPS_EVAL + 1:
            pred_invalid_count += 1

        # 1-step / k-step MAE over valid pairs.
        n_pairs = min(len(pred_traj) - 1, len(true_traj) - 1)
        if n_pairs > 0:
            u_pairs = true_traj[:n_pairs]
            y_true = true_traj[1 : n_pairs + 1] - u_pairs
            valid_pairs = np.isfinite(u_pairs).all(axis=1) & np.isfinite(y_true).all(axis=1)
            if np.any(valid_pairs):
                u_pairs = u_pairs[valid_pairs]
                y_true = y_true[valid_pairs]
                X = np.concatenate(
                    [u_pairs, np.repeat(p[None, :], len(u_pairs), axis=0)],
                    axis=1,
                )
                y_pred = model.predict(X)
                valid_pred = np.isfinite(y_pred).all(axis=1)
                if np.any(valid_pred):
                    y_pred = y_pred[valid_pred]
                    y_true = y_true[valid_pred]
                    one_step_abs.append(np.mean(np.abs(y_pred - y_true)))

        # k-step state MAE: compare u_{t+k} from model rollout and true rollout.
        m_state = min(len(pred_traj), len(true_traj))
        for k in K_STEPS:
            if m_state > k:
                pred_k = pred_traj[k:m_state]
                true_k = true_traj[k:m_state]
                valid_k = np.isfinite(pred_k).all(axis=1) & np.isfinite(true_k).all(axis=1)
                if np.any(valid_k):
                    k_step_abs[k].append(np.mean(np.abs(pred_k[valid_k] - true_k[valid_k])))

        # Rollout RMSE over common prefix.
        m = min(len(pred_traj), len(true_traj))
        if m > 0:
            diff = pred_traj[:m] - true_traj[:m]
            valid_states = np.isfinite(diff).all(axis=1)
            if np.any(valid_states):
                diff = diff[valid_states]
                rollout_sqerr.append(np.sum(diff * diff))
                rollout_true_sq.append(np.sum(true_traj[:m][valid_states] ** 2))
                rollout_points += diff.size

    one_step_mae = float(np.mean(one_step_abs)) if one_step_abs else float("nan")
    rollout_rmse = float(np.sqrt(np.sum(rollout_sqerr) / rollout_points)) if rollout_points > 0 else float("nan")
    rollout_nrmse = (
        float(np.sqrt(np.sum(rollout_sqerr) / np.sum(rollout_true_sq)))
        if np.sum(rollout_true_sq) > 0
        else float("nan")
    )
    divergence_rate = pred_diverged_count / len(params) if len(params) > 0 else float("nan")
    true_invalid_rate = true_invalid_count / len(params) if len(params) > 0 else float("nan")
    pred_invalid_rate = pred_invalid_count / len(params) if len(params) > 0 else float("nan")

    out = {
        "one_step_mae": one_step_mae,
        "rollout_rmse": rollout_rmse,
        "rollout_nrmse": rollout_nrmse,
        "divergence_rate": divergence_rate,
        "true_invalid_rate": true_invalid_rate,
        "pred_invalid_rate": pred_invalid_rate,
    }
    for k in K_STEPS:
        out[f"k{k}_mae"] = float(np.mean(k_step_abs[k])) if k_step_abs[k] else float("nan")
    return out


def fixed_point_proxy(model, params: np.ndarray) -> dict:
    counts = []
    for p in tqdm(params, total=len(params), desc="Fixed-point proxy", unit="param", leave=False):
        fps = collect_fixed_points_trajectory_grid(
            model=model,
            p=p,
            u_start_axes=FP_AXIS,
            max_iter=FP_MAX_ITER,
            d_tol=FP_D_TOL,
            unique_tol=FP_UNIQUE_TOL,
            divergence_threshold=ROLLOUT_DIVERGENCE_THRESHOLD,
            verify_residual_tol=None,
        )
        counts.append(len(fps))
    return {
        "fixed_points_mean": float(np.mean(counts)),
        "fixed_points_std": float(np.std(counts)),
    }


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


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    initials_id = sample_initial_conditions(rng, N_TRAJ_EVAL)
    params_id = sample_params(rng, N_TRAJ_EVAL, PARAMETERS_RANGES_ID)

    initials_ood = sample_initial_conditions(rng, N_TRAJ_EVAL)
    params_ood = sample_params(rng, N_TRAJ_EVAL, PARAMETERS_RANGES_OOD)

    fp_params_id = sample_params(rng, FP_PARAM_SAMPLES, PARAMETERS_RANGES_ID)
    fp_params_ood = sample_params(rng, FP_PARAM_SAMPLES, PARAMETERS_RANGES_OOD)

    rows = []
    for model_name in tqdm(
        ["original", "target_normalized", "manuscript"],
        desc="Evaluate models",
        unit="model",
    ):
        print(f"Evaluating {model_name}...")
        model = load_model(model_name)

        id_metrics = evaluate_split(model, params_id, initials_id)
        ood_metrics = evaluate_split(model, params_ood, initials_ood)
        fp_id = fixed_point_proxy(model, fp_params_id)
        fp_ood = fixed_point_proxy(model, fp_params_ood)

        row = {
            "model": model_name,
            "id_one_step_mae": id_metrics["one_step_mae"],
            "id_k1_mae": id_metrics["k1_mae"],
            "id_k5_mae": id_metrics["k5_mae"],
            "id_k10_mae": id_metrics["k10_mae"],
            "id_k30_mae": id_metrics["k30_mae"],
            "id_rollout_rmse": id_metrics["rollout_rmse"],
            "id_rollout_nrmse": id_metrics["rollout_nrmse"],
            "id_divergence_rate": id_metrics["divergence_rate"],
            "id_true_invalid_rate": id_metrics["true_invalid_rate"],
            "id_pred_invalid_rate": id_metrics["pred_invalid_rate"],
            "ood_one_step_mae": ood_metrics["one_step_mae"],
            "ood_k1_mae": ood_metrics["k1_mae"],
            "ood_k5_mae": ood_metrics["k5_mae"],
            "ood_k10_mae": ood_metrics["k10_mae"],
            "ood_k30_mae": ood_metrics["k30_mae"],
            "ood_rollout_rmse": ood_metrics["rollout_rmse"],
            "ood_rollout_nrmse": ood_metrics["rollout_nrmse"],
            "ood_divergence_rate": ood_metrics["divergence_rate"],
            "ood_true_invalid_rate": ood_metrics["true_invalid_rate"],
            "ood_pred_invalid_rate": ood_metrics["pred_invalid_rate"],
            "id_fixed_points_mean": fp_id["fixed_points_mean"],
            "ood_fixed_points_mean": fp_ood["fixed_points_mean"],
            "id_fixed_points_std": fp_id["fixed_points_std"],
            "ood_fixed_points_std": fp_ood["fixed_points_std"],
        }
        rows.append(row)
        print(row)

    csv_path = RESULTS_DIR / "metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = RESULTS_DIR / "metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    print(f"Saved metrics: {csv_path}")
    print(f"Saved metrics: {json_path}")


if __name__ == "__main__":
    main()
