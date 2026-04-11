"""
Обучение NeuroMap на Чуа с отбрасыванием траекторий, чья амплитуда на сечении
S = y не ниже REJECT_AMPLITUDE_ABOVE (см. e2_chua_config.py и ноутбук подбора порога).
"""
import sys
from pathlib import Path

import numpy as np

_M = Path(__file__).resolve().parent
_REPO_ROOT = _M.parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_M))

from utils import DynamicSystemDatasetGenerator
from neuromaps import NeuroMapManuscript
from systems.chua_rk4 import chua_rk4, chua_right_part

from e2_chua_config import (
    SEED,
    VARIABLES_RANGES,
    PARAMETERS_RANGES,
    NUM_OF_TRAJ,
    NUM_IN_TRAJ,
    DT,
    HIDDEN_SIZE,
    NUM_HIDDEN_LAYERS,
    EPOCHS,
    LR,
    BATCH_SIZE,
    VAL_SPLIT,
    LR_SCHEDULER,
    LR_SCHEDULER_PATIENCE,
    LR_SCHEDULER_FACTOR,
    VAL_EVERY,
    N_TRANSIENT,
    DATASET_EXCLUDE_LARGE_NPZ,
    REJECT_AMPLITUDE_ABOVE,
)

if REJECT_AMPLITUDE_ABOVE is None:
    raise ValueError(
        "Задайте REJECT_AMPLITUDE_ABOVE в e2_chua_config.py "
        "(подбор в amplitude_threshold_sweep.ipynb)."
    )


def secant_plane(state, params):
    _ = params
    return float(state[1])


def secant_plane_derivatives(state, params):
    _ = state, params
    return np.array([0.0, 1.0, 0.0], dtype=np.float64)


CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints" / "exclude_large_lc"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

target_samples = NUM_OF_TRAJ * NUM_IN_TRAJ

print(
    f"Генерация данных (исключаем амплитуду >= {REJECT_AMPLITUDE_ABOVE})…"
)
generator = DynamicSystemDatasetGenerator(
    evolution_operator=chua_rk4,
    right_part=chua_right_part,
    variables_ranges=VARIABLES_RANGES,
    parameters_ranges=PARAMETERS_RANGES,
    n_transient=N_TRANSIENT,
    steps_per_trajectory=NUM_IN_TRAJ,
    secant_plane=secant_plane,
    secant_plane_derivatives=secant_plane_derivatives,
    dt=DT,
    seed=SEED,
    reject_amplitude_above=REJECT_AMPLITUDE_ABOVE,
)
X, y, info = generator.generate(target_samples=target_samples, n_jobs=-1)

dataset_path = _M / DATASET_EXCLUDE_LARGE_NPZ
dataset_path.parent.mkdir(parents=True, exist_ok=True)
generator.save(str(dataset_path), overwrite=True)
print(f"Датасет сохранён: {dataset_path}")

print("X stats: min =", X.min(), "max =", X.max(), "mean abs =", np.mean(np.abs(X)))
print("y stats: min =", y.min(), "max =", y.max(), "mean abs =", np.mean(np.abs(y)))
print(
    "EP rejected =",
    info.get("rejected_fixed_points"),
    "| large amplitude rejected =",
    info.get("rejected_large_amplitude"),
    "| accepted trajectories =",
    info.get("accepted_trajectories"),
    "| total processed =",
    info.get("total_trajectories_processed"),
)

checkpoint_files = sorted(
    CHECKPOINT_DIR.glob("epoch=*.ckpt"),
    key=lambda p: p.stat().st_mtime,
)

if checkpoint_files:
    latest_ckpt = checkpoint_files[-1]
    print(f"Найден чекпоинт для продолжения: {latest_ckpt}")
    model = NeuroMapManuscript.load(str(latest_ckpt))
else:
    print("Создаём новую модель…")
    model = NeuroMapManuscript(
        n_var=3,
        n_param=5,
        hidden_size=HIDDEN_SIZE,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        dt=DT,
    )

print("Запускаем обучение…")
model.fit(
    X,
    y,
    epochs=EPOCHS,
    lr=LR,
    batch_size=BATCH_SIZE,
    val_split=VAL_SPLIT,
    checkpoint_dir=str(CHECKPOINT_DIR),
    history_path=str(CHECKPOINT_DIR / "history.json"),
    lr_scheduler=LR_SCHEDULER,
    lr_scheduler_patience=LR_SCHEDULER_PATIENCE,
    lr_scheduler_factor=LR_SCHEDULER_FACTOR,
    val_every=VAL_EVERY,
)

final_path = CHECKPOINT_DIR / "model.ckpt"
print(f"Сохраняем финальную модель в {final_path}")
model.save(str(final_path))
print("Обучение завершено успешно!")
