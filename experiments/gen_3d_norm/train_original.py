import numpy as np
from pathlib import Path

from systems.generator_3d import generator_3d_rk4, generator_3d_right_part
from utils import DynamicSystemDatasetGenerator
from neuromaps import NeuroMapOriginal

dataset_generator = DynamicSystemDatasetGenerator(
    evolution_operator=generator_3d_rk4,
    right_part=generator_3d_right_part,
    variables_ranges=[(-10, 10), (-50, 50), (-20, 20)],
    parameters_ranges=[(-1, 7), (1/23, 1/18), (5, 8), (0.02, 0.02)], # lambda_, beta, w0, k 
    n_transient=250,
    steps_per_trajectory=5,
    fp_threshold=1e-4,
    div_threshold=750,
    secant_plane=lambda x, y: x[1],
    secant_plane_derivatives=lambda x, y: [0, 1, 0],
    accuracy=1e-6,
    dt=0.01,
    seed=52,
    n_jobs=50
)
X, y, info = dataset_generator.generate(target_samples=100_000, n_jobs=20)

dataset_generator.save("experiments/gen_3d_norm/dataset.npz", overwrite=True)

print(info)

checkpoint_dir = "experiments/gen_3d_norm/checkpoints/original"
checkpoint_dir_path = Path(checkpoint_dir)

checkpoint_files = list(checkpoint_dir_path.glob("epoch=*.ckpt"))
if checkpoint_files:
    latest_checkpoint = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
    print(f"Найден чекпоинт для продолжения: {latest_checkpoint}")
    print("Модель будет загружена автоматически при обучении")
    model = NeuroMapOriginal(n_var=3, n_param=4, hidden_size=256, dt=0.01)
else:
    checkpoint_path = checkpoint_dir_path / "model.ckpt"
    if checkpoint_path.exists():
        print(f"Загружаем модель из {checkpoint_path}")
        model = NeuroMapOriginal.load(str(checkpoint_path))
        print("Модель загружена, продолжаем обучение...")
    else:
        print("Создаем новую модель")
        model = NeuroMapOriginal(n_var=3, n_param=4, hidden_size=256, dt=0.01)

model.fit(X, y, epochs=2000, lr=1e-4, batch_size=126, val_split=0.2, 
          checkpoint_dir=checkpoint_dir,
          history_path="experiments/gen_3d_norm/checkpoints/original/history.json")

model_path = "experiments/gen_3d_norm/checkpoints/original/model.ckpt"
model.save(model_path)
print(f"Модель сохранена в {model_path}")
