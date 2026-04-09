import numpy as np
from pathlib import Path
import torch

from systems.generator_3d import generator_3d_rk4
from utils import generate_pairs_dataset
from neuromaps import NeuroMapOriginal

def main() -> None:
    torch.manual_seed(52)
    np.random.seed(52)

    X, y = generate_pairs_dataset(
        evolution_operator=generator_3d_rk4,
        variables_ranges=[(-10, 10), (-50, 50), (-20, 20)],
        parameters_ranges=[(-1, 7), (1 / 23, 1 / 18), (5, 8), (0.02, 0.02)],  # lambda_, beta, w0, k
        num_of_traj=200_000,
        num_in_traj=5,
        dt=0.001,
        seed=52,
    )

    checkpoint_dir = "experiments/gen_3d_norm/checkpoints/original"
    history_path = "experiments/gen_3d_norm/checkpoints/original/history.json"
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    print("Подготовка модели...")
    model = NeuroMapOriginal(n_var=3, n_param=4, hidden_size=256, dt=0.001, lr=1e-4)

    print("\n=== Начало обучения ===")
    model.fit(
        X,
        y,
        epochs=2000,
        lr=1e-4,
        batch_size=128,
        val_split=0.2,
        val_every=1,
        log_every=50,
        verbose=True,
        num_workers=0,  # safer default on macOS/Python 3.12
        checkpoint_dir=checkpoint_dir,
        history_path=history_path,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        early_stopping_patience=100,
        lr_scheduler=True,
        lr_scheduler_patience=30,
        lr_scheduler_factor=0.5,
        ckpt_path=None,
    )

    model_path = "experiments/gen_3d_norm/checkpoints/original/model_final.ckpt"
    model.save(model_path, save_history=True)
    print(f"\nМодель успешно сохранена в {model_path}")

    if hasattr(model, "training_history"):
        best_val_loss = min(model.training_history["val_loss"])
        best_epoch = model.training_history["val_loss"].index(best_val_loss)
        print(f"Лучший результат на валидации: {best_val_loss:.6f} на эпохе {best_epoch+1}")


if __name__ == "__main__":
    main()
