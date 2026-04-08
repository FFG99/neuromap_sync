import numpy as np
from pathlib import Path

from systems import chua_rk4
from utils import generate_pairs_dataset
from neuromaps import NeuroMapTargetNormalized

X, y = generate_pairs_dataset(
    evolution_operator=chua_rk4,
    variables_ranges=[(-15, 15), (-4, 4), (-18, 18)],
    parameters_ranges=[(8.4, 8.4), (12, 12), (0, 0), (-0.12, -0.12), (-1.15, -1.15)],
    num_of_traj=20000,
    num_in_traj=5,
    dt=0.01,
    seed=52
)

checkpoint_dir = "experiments/normalization_chua/checkpoints/fixed"
checkpoint_dir_path = Path(checkpoint_dir)

checkpoint_files = list(checkpoint_dir_path.glob("epoch=*.ckpt"))
if checkpoint_files:
    latest_checkpoint = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
    print(f"Найден чекпоинт для продолжения: {latest_checkpoint}")
    print("Модель будет загружена автоматически при обучении")
    model = NeuroMapTargetNormalized(n_var=3, n_param=5, hidden_size=256, dt=0.01)
else:
    checkpoint_path = checkpoint_dir_path / "model.ckpt"
    if checkpoint_path.exists():
        print(f"Загружаем модель из {checkpoint_path}")
        model = NeuroMapTargetNormalized.load(str(checkpoint_path))
        print("Модель загружена, продолжаем обучение...")
    else:
        print("Создаем новую модель")
        model = NeuroMapTargetNormalized(n_var=3, n_param=5, hidden_size=256, dt=0.01)

model.fit(X, y, epochs=1000, lr=5e-4, batch_size=512, val_split=0.2, 
          checkpoint_dir=checkpoint_dir,
          history_path="experiments/normalization_chua/checkpoints/fixed/history.json")

model_path = "experiments/normalization_chua/checkpoints/fixed/model.ckpt"
model.save(model_path)
print(f"Модель сохранена в {model_path}")
