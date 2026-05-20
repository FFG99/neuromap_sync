#!/usr/bin/env python3
"""
Обучение 4 вариантов NeuroMapManuscriptLR (2×2: подсети × Z-score).

Варианты (чекпоинты в checkpoints/<name>/):
  - subnets_none, subnets_zscore
  - shared_none, shared_zscore

Датасет генерируется один раз и кэшируется в datasets/e1_vdp_mod1.npz.

Запуск из корня репозитория::

    python experiments/manuscript_lr/e1_train.py
    python experiments/manuscript_lr/e1_train.py --variant subnets_zscore
    python experiments/manuscript_lr/e1_train.py --regenerate-data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from experiments.manuscript_lr.e1_config import (
    BATCH_SIZE,
    DATASET_PATH,
    DT,
    EPOCHS,
    HIDDEN_SIZE,
    LR,
    LR_SCHEDULER_FACTOR,
    LR_SCHEDULER_PATIENCE,
    N_PARAM,
    N_VAR,
    NUM_HIDDEN_LAYERS,
    NUM_IN_TRAJ,
    NUM_OF_TRAJ,
    PARAMETERS_RANGES,
    SEED,
    TRAINING_VARIANTS,
    VAL_SPLIT,
    VARIABLES_RANGES,
    VARIANT_NAMES,
    checkpoint_dir,
    model_ckpt_path,
    variant_by_name,
)
from neuromaps.nm_manuscript_lr import NeuroMapManuscriptLR
from systems.vdp_mod1 import vdp_mod1_rk4
from utils import generate_pairs_dataset_finite


def load_or_generate_dataset(regenerate: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает (X, y_next)."""
    if DATASET_PATH.is_file() and not regenerate:
        data = np.load(DATASET_PATH)
        X = data["X"]
        y_next = data["y_next"] if "y_next" in data else data["X"][:, :N_VAR] + data["y"]
        print(f"Загружен кэш датасета: {DATASET_PATH}  X={X.shape}")
        return X, y_next

    print(f"Генерация данных: {NUM_OF_TRAJ:,} траекторий × {NUM_IN_TRAJ} шагов…")
    X, y_delta = generate_pairs_dataset_finite(
        evolution_operator=vdp_mod1_rk4,
        variables_ranges=VARIABLES_RANGES,
        parameters_ranges=PARAMETERS_RANGES,
        num_of_traj=NUM_OF_TRAJ,
        num_in_traj=NUM_IN_TRAJ,
        dt=DT,
        seed=SEED,
    )
    y_next = X[:, :N_VAR] + y_delta
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DATASET_PATH, X=X, y_delta=y_delta, y_next=y_next)
    print(f"Сохранён кэш: {DATASET_PATH}")
    print(
        "X:", X.shape,
        "min =", X.min(), "max =", X.max(), "mean |x| =", np.mean(np.abs(X)),
    )
    print(
        "Δu:", y_delta.shape,
        "min =", y_delta.min(), "max =", y_delta.max(), "mean |Δu| =", np.mean(np.abs(y_delta)),
    )
    return X, y_next


def train_variant(
    variant: dict,
    X: np.ndarray,
    y_next: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    skip_if_done: bool,
) -> None:
    name = variant["name"]
    ckpt_dir = checkpoint_dir(name)
    final_path = model_ckpt_path(name)

    print("\n" + "=" * 72)
    print(
        f"Вариант: {name}  |  use_subnets={variant['use_subnets']}  |  "
        f"norm_mode={variant['norm_mode']!r}"
    )
    print("=" * 72)

    if skip_if_done and final_path.is_file():
        print(f"Пропуск: уже есть {final_path}")
        return

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    epoch_ckpts = sorted(ckpt_dir.glob("epoch=*.ckpt"), key=lambda p: p.stat().st_mtime)

    if epoch_ckpts:
        latest = epoch_ckpts[-1]
        print(f"Продолжение с чекпоинта: {latest}")
        model = NeuroMapManuscriptLR.load(str(latest))
    else:
        print("Новая модель…")
        model = NeuroMapManuscriptLR(
            n_var=N_VAR,
            n_param=N_PARAM,
            hidden_size=HIDDEN_SIZE,
            num_hidden_layers=NUM_HIDDEN_LAYERS,
            dt=DT,
            lr=lr,
            use_subnets=variant["use_subnets"],
            norm_mode=variant["norm_mode"],
        )

    model.fit(
        X,
        y_next,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        val_split=VAL_SPLIT,
        checkpoint_dir=str(ckpt_dir),
        history_path=str(ckpt_dir / "history.json"),
        lr_scheduler=True,
        lr_scheduler_patience=LR_SCHEDULER_PATIENCE,
        lr_scheduler_factor=LR_SCHEDULER_FACTOR,
        val_every=1,
    )

    model.save(str(final_path))
    print(f"Сохранено: {final_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--variant",
        choices=VARIANT_NAMES,
        action="append",
        help="Обучить только указанные варианты (можно повторить флаг). По умолчанию — все 4.",
    )
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument(
        "--regenerate-data",
        action="store_true",
        help="Пересоздать datasets/e1_vdp_mod1.npz",
    )
    p.add_argument(
        "--skip-if-done",
        action="store_true",
        help="Пропустить вариант, если checkpoints/<name>/model.ckpt уже есть",
    )
    args = p.parse_args()

    np.random.seed(SEED)
    X, y_next = load_or_generate_dataset(regenerate=args.regenerate_data)

    if args.variant:
        variants = [variant_by_name(n) for n in args.variant]
    else:
        variants = list(TRAINING_VARIANTS)

    print(f"Будет обучено вариантов: {len(variants)}")
    for v in variants:
        train_variant(
            v,
            X,
            y_next,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            skip_if_done=args.skip_if_done,
        )

    print("\nВсе запрошенные варианты завершены.")
    for v in variants:
        print(f"  {v['name']}: {model_ckpt_path(v['name'])}")


if __name__ == "__main__":
    main()
