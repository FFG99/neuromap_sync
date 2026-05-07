#!/usr/bin/env python3
from pathlib import Path

import numpy as np

from neuromaps import NeuroMapManuscriptSubnets

ARTIFACTS_DIR = Path("experiments/manuscript_0303_exact/artifacts/model4")
CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints_subnets"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

DT = 0.01
EPOCHS = 1000
LR = 1e-3
BATCH_SIZE = 256
VAL_SPLIT = 0.2
NUM_HIDDEN_LAYERS = 1


def load_dataset(path: Path):
    data = np.load(path)
    return data["X"], data["y"]


def main() -> None:
    train_path = ARTIFACTS_DIR / "train_dataset.npz"
    val_path = ARTIFACTS_DIR / "val_dataset.npz"
    if not train_path.is_file() or not val_path.is_file():
        raise FileNotFoundError("Run 11_generate_dataset_model4.py first.")

    X_train, y_train = load_dataset(train_path)
    X_val, y_val = load_dataset(val_path)
    X = np.concatenate([X_train, X_val], axis=0)
    y = np.concatenate([y_train, y_val], axis=0)

    checkpoint_files = sorted(CHECKPOINT_DIR.glob("epoch=*.ckpt"), key=lambda p: p.stat().st_mtime)
    if checkpoint_files:
        latest_ckpt = checkpoint_files[-1]
        print(f"Resume from checkpoint: {latest_ckpt}")
        model = NeuroMapManuscriptSubnets.load(str(latest_ckpt))
    else:
        print("Create NeuroMapManuscriptSubnets")
        model = NeuroMapManuscriptSubnets(
            n_var=2,
            n_param=2,
            hidden_size=100,
            num_hidden_layers=NUM_HIDDEN_LAYERS,
            dt=DT,
        )

    model.fit(
        X,
        y,
        epochs=EPOCHS,
        lr=LR,
        batch_size=BATCH_SIZE,
        val_split=VAL_SPLIT,
        checkpoint_dir=str(CHECKPOINT_DIR),
        history_path=str(CHECKPOINT_DIR / "history.json"),
        lr_scheduler=False,
        val_every=1,
    )

    final_model_path = CHECKPOINT_DIR / "model.ckpt"
    model.save(str(final_model_path))
    print(f"Saved model: {final_model_path}")


if __name__ == "__main__":
    main()
