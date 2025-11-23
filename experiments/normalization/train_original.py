import numpy as np

from systems.van_der_pol_rk4 import van_der_pol_rk4
from utils import generate_pairs_dataset
from neuromaps import NeuroMapOriginal

X, y = generate_pairs_dataset(
    evolution_operator=van_der_pol_rk4,
    variables_ranges=[(-10, 10), (-100, 100)],
    parameters_ranges=[(-10, 10), (0, 10)],
    num_of_traj=20000,
    num_in_traj=5,
    dt=0.01,
    seed=52
)

model = NeuroMapOriginal(n_var=2, n_param=2, hidden_size=128, dt=0.01)
model.fit(X, y, epochs=1000, lr=1e-3, batch_size=256, val_split=0.1, 
          checkpoint_dir="experiments/single_vdp_nm_1/checkpoints",
          history_path="experiments/single_vdp_nm_1/checkpoints/history.json")

model_path = "experiments/single_vdp_nm_1/checkpoints/model_original.ckpt"
model.save(model_path)
print(f"Модель сохранена в {model_path}")
