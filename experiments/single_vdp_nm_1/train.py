import numpy as np

from systems.van_der_pol_rk4 import van_der_pol_rk4
from utils import generate_pairs_dataset

X, y = generate_pairs_dataset(
    evolution_operator=van_der_pol_rk4,
    variables_ranges=[(-10, 10), (-100, 100)],
    parameters_ranges=[(-10, 10), (0, 10)],
    num_of_traj=10000,
    num_in_traj=5,
    dt=0.01,
    seed=42
)

from neuromaps import NeuroMap1
from pathlib import Path

model = NeuroMap1(n_var=2, n_param=2, hidden_size=128, dt=0.01)

model.fit(X, y, epochs=100, lr=1e-3, batch_size=256, val_split=0.1)

# Сохранение модели
model_path = Path(__file__).parent / "model.ckpt"
model.save(model_path)
print(f"Модель сохранена в {model_path}")

# model = NeuroMap1.load("experiments/single_vdp_nm_1/model.ckpt")
