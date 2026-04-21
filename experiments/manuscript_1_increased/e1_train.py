import os
import numpy as np
from pathlib import Path

import torch

from utils import generate_pairs_dataset_finite
from neuromaps import NeuroMapManuscript

from systems.vdp_mod1 import vdp_mod1_rk4

SEED = 42

VARIABLES_RANGES = [(-10.19, 10.18), (-136.5, 136.5)]   # x₁, x₂
PARAMETERS_RANGES = [(-3.0, 1.0), (0.02, 0.1)]  # λ , β

NUM_OF_TRAJ = 125_000
NUM_IN_TRAJ = 10
DT          = 0.01

print("Генерация данных…")
X, y = generate_pairs_dataset_finite(
    evolution_operator=vdp_mod1_rk4,
    variables_ranges=VARIABLES_RANGES,
    parameters_ranges=PARAMETERS_RANGES,
    num_of_traj=NUM_OF_TRAJ,
    num_in_traj=NUM_IN_TRAJ,
    dt=DT,
    seed=SEED
)
print("X stats: min =", X.min(), "max =", X.max(), "mean abs =", np.mean(np.abs(X)))
print("y stats: min =", y.min(), "max =", y.max(), "mean abs =", np.mean(np.abs(y)))


CHECKPOINT_DIR = Path("experiments/manuscript_1_increased/checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

checkpoint_files = sorted(
    CHECKPOINT_DIR.glob("epoch=*.ckpt"),
    key=lambda p: p.stat().st_mtime
)

if checkpoint_files:
    latest_ckpt = checkpoint_files[-1]
    print(f"Найден чекпоинт для продолжения: {latest_ckpt}")
    model = NeuroMapManuscript.load(str(latest_ckpt))
else:
    print("Создаём новую модель…")
    model = NeuroMapManuscript(
        n_var=2,
        n_param=2,
        hidden_size=100,
        num_hidden_layers=2,
        dt=DT
    )

print("Запускаем обучение…")
model.fit(
    X, y,
    epochs=1000,
    lr=1e-3,
    batch_size=256,
    val_split=0.2,
    checkpoint_dir=str(CHECKPOINT_DIR),
    history_path=str(CHECKPOINT_DIR / "history.json"),
    lr_scheduler=True,
    lr_scheduler_patience=10,
    lr_scheduler_factor=0.1,
    val_every=1
)

FINAL_MODEL_PATH = CHECKPOINT_DIR / "model.ckpt"
print(f"Сохраняем финальную модель в {FINAL_MODEL_PATH}")
model.save(str(FINAL_MODEL_PATH))

print("Обучение завершено успешно!")
