import numpy as np
from pathlib import Path

from systems.van_der_pol_rk4 import van_der_pol_rk4
from utils import generate_pairs_dataset
from neuromaps import NeuroMapOriginal

X, y = generate_pairs_dataset(
    evolution_operator=van_der_pol_rk4,
    variables_ranges=[(-10, 10), (-100, 100)],
    parameters_ranges=[(-10, 10), (0, 10)],
    num_of_traj=200000,
    num_in_traj=5,
    dt=0.01,
    seed=52
)

checkpoint_dir = "experiments/normalization_2layers/checkpoints/original"
checkpoint_dir_path = Path(checkpoint_dir)

# Ищем последний чекпоинт с паттерном epoch=*.ckpt
checkpoint_files = list(checkpoint_dir_path.glob("epoch=*.ckpt"))
if checkpoint_files:
    # Используем последний чекпоинт (по времени модификации)
    latest_checkpoint = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
    print(f"Найден чекпоинт для продолжения: {latest_checkpoint}")
    print("Модель будет загружена автоматически при обучении")
    model = NeuroMapOriginal(n_var=2, n_param=2, hidden_size=256, num_hidden_layers=2, dt=0.01)
else:
    # Пытаемся загрузить из model.ckpt, если он существует
    checkpoint_path = checkpoint_dir_path / "model.ckpt"
    if checkpoint_path.exists():
        print(f"Загружаем модель из {checkpoint_path}")
        model = NeuroMapOriginal.load(str(checkpoint_path))
        print("Модель загружена, продолжаем обучение...")
    else:
        print("Создаем новую модель (2 скрытых слоя)")
        model = NeuroMapOriginal(n_var=2, n_param=2, hidden_size=256, num_hidden_layers=2, dt=0.01)

model.fit(X, y, epochs=1000, lr=5e-4, batch_size=512, val_split=0.2,
          checkpoint_dir=checkpoint_dir,
          history_path="experiments/normalization_2layers/checkpoints/original/history.json")

model_path = "experiments/normalization_2layers/checkpoints/original/model.ckpt"
model.save(model_path)
print(f"Модель сохранена в {model_path}")
