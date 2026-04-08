import numpy as np
from pathlib import Path

from systems.van_der_pol_rk4 import van_der_pol_rk4
from utils import generate_pairs_dataset
from neuromaps import NeuroMapTargetNormalized

X, y = generate_pairs_dataset(
    evolution_operator=van_der_pol_rk4,
    variables_ranges=[(-10, 10), (-100, 100)],
    parameters_ranges=[(-10, 10), (0, 10)],
    num_of_traj=1_000_000,
    num_in_traj=5,
    dt=0.01,
    seed=52
)

model = NeuroMapTargetNormalized(
    n_var=2,
    n_param=2,
    hidden_size=1024,
    dt=0.01
)

model.fit(
    X,
    y,
    epochs=1000,
    lr=1e-4,
    batch_size=512,
    val_split=0.2,
    checkpoint_dir="experiments/increased_mlp/checkpoints",
    history_path="experiments/increased_mlp/checkpoints/history.json"
)

model_path = "experiments/increased_mlp/checkpoints/model.ckpt"
model.save(model_path)
print(f"Модель сохранена в {model_path}")
