"""
Обучение NeuroMap на vdp_mod2: полная область PARAMETERS_RANGES из e2_config (без вырезаний).
"""
import sys
from pathlib import Path

import numpy as np

_M2 = Path(__file__).resolve().parent
_REPO_ROOT = _M2.parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_M2))

from utils import generate_pairs_dataset_finite
from neuromaps import NeuroMapManuscript
from systems.vdp_mod2 import vdp_mod2_rk4

from e2_config import (
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
)

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints" / "mu_full"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

print("Генерация данных (полная область по μ)…")
X, y = generate_pairs_dataset_finite(
    evolution_operator=vdp_mod2_rk4,
    variables_ranges=VARIABLES_RANGES,
    parameters_ranges=PARAMETERS_RANGES,
    num_of_traj=NUM_OF_TRAJ,
    num_in_traj=NUM_IN_TRAJ,
    dt=DT,
    seed=SEED,
)
print("X stats: min =", X.min(), "max =", X.max(), "mean abs =", np.mean(np.abs(X)))
print("y stats: min =", y.min(), "max =", y.max(), "mean abs =", np.mean(np.abs(y)))

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
        n_var=2,
        n_param=2,
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
