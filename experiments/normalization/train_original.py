import numpy as np

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

model = NeuroMapOriginal(n_var=2, n_param=2, hidden_size=256, dt=0.01)
model.fit(X, y, epochs=500, lr=5e-4, batch_size=512, val_split=0.2, 
          checkpoint_dir="experiments/normalization/checkpoints/original",
          history_path="experiments/normalization/checkpoints/original/history.json")

model_path = "experiments/normalization/checkpoints/original/model.ckpt"
model.save(model_path)
print(f"Модель сохранена в {model_path}")
