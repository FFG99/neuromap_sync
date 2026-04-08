from systems import chua_rk4
from utils import generate_sequence_dataset
from neuromaps import NeuroMapTargetNormalized

X_seq, y_seq = generate_sequence_dataset(
    evolution_operator=chua_rk4,
    variables_ranges=[(-15, 15), (-4, 4), (-18, 18)],
    parameters_ranges=[(8.4, 8.4), (12, 12), (0, 0),
                       (-0.12, -0.12), (-1.15, -1.15)],
    num_of_traj=10000,
    seq_len=20, 
    dt=0.01,
    seed=52
)

model = NeuroMapTargetNormalized(n_var=3, n_param=5, hidden_size=512, dt=0.01)

model.fit_recursive(
    X_seq, y_seq,
    epochs=1000,
    lr=5e-4,
    batch_size=1024,
    val_split=0.2,
    val_every=10,
    log_every=50,
    checkpoint_dir="experiments/long_trajs_chua/checkpoints/long",
    history_path="experiments/long_trajs_chua/checkpoints/long/history.json"
)

model_path = "experiments/long_trajs_chua/checkpoints/long/model.ckpt"
model.save(model_path)
print(f"Модель сохранена в {model_path}")
