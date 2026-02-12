from systems import chua_rk4
from utils import generate_sequence_dataset
from neuromaps import NeuroMapFixed
import numpy as np

# Генерируем тот же датасет последовательностей
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

# Преобразуем последовательности в пары (X, y) для обычного обучения
# X_seq: (N, seq_len, n_var + n_param)
# y_seq: (N, seq_len, n_var)
# Нужно получить X: (N*seq_len, n_var + n_param), y: (N*seq_len, n_var)
N, seq_len = X_seq.shape[0], X_seq.shape[1]
X = X_seq.reshape(N * seq_len, -1)  # (N*seq_len, n_var + n_param)
y = y_seq.reshape(N * seq_len, -1)  # (N*seq_len, n_var)

print(f"Датасет преобразован: X.shape={X.shape}, y.shape={y.shape}")

model = NeuroMapFixed(n_var=3, n_param=5, hidden_size=512, dt=0.01)

model.fit(
    X, y,
    epochs=1000,
    lr=5e-4,
    batch_size=1024,
    val_split=0.2,
    val_every=10,
    log_every=50,
    checkpoint_dir="experiments/long_trajs_chua/checkpoints/short",
    history_path="experiments/long_trajs_chua/checkpoints/short/history.json"
)

model_path = "experiments/long_trajs_chua/checkpoints/short/model.ckpt"
model.save(model_path)
print(f"Модель сохранена в {model_path}")
