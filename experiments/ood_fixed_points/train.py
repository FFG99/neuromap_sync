from pathlib import Path
import json
from typing import Optional

import numpy as np
from tqdm import tqdm

from systems.vdp_mod1 import vdp_mod1_rk4
from utils import generate_pairs_dataset_finite
from neuromaps import (
    NeuroMapOriginal,
    NeuroMapTargetNormalized,
    NeuroMapManuscript,
)


SEED = 123
DT = 0.01

# Research-grade setup (can be increased later).
NUM_OF_TRAJ = 12_500
NUM_IN_TRAJ = 10
EPOCHS = 300
BATCH_SIZE = 512
VAL_SPLIT = 0.2

VARIABLES_RANGES = [(-8.0, 8.0), (-80.0, 80.0)]
PARAMETERS_RANGES_TRAIN = [(-2.0, 0.4), (0.03, 0.08)]  # lambda, beta

ROOT_DIR = Path("experiments/ood_fixed_points")
CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"
CONFIG_PATH = ROOT_DIR / "config.json"


def latest_epoch_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    if not ckpt_dir.exists():
        return None
    files = sorted(ckpt_dir.glob("epoch=*.ckpt"), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def ensure_dirs() -> None:
    ROOT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ensure_dirs()

    print("Generating training pairs...")
    X, y = generate_pairs_dataset_finite(
        evolution_operator=vdp_mod1_rk4,
        variables_ranges=VARIABLES_RANGES,
        parameters_ranges=PARAMETERS_RANGES_TRAIN,
        num_of_traj=NUM_OF_TRAJ,
        num_in_traj=NUM_IN_TRAJ,
        dt=DT,
        seed=SEED,
    )
    print(f"Dataset: X={X.shape}, y={y.shape}")

    models = {
        "original": NeuroMapOriginal,
        "target_normalized": NeuroMapTargetNormalized,
        "manuscript": NeuroMapManuscript,
    }

    for model_name, model_cls in tqdm(
        models.items(), total=len(models), desc="Training models", unit="model"
    ):
        model_ckpt_dir = CHECKPOINTS_DIR / model_name
        model_ckpt_dir.mkdir(parents=True, exist_ok=True)

        ckpt = latest_epoch_checkpoint(model_ckpt_dir)
        if ckpt is not None:
            print(f"[{model_name}] Resume from {ckpt}")
            model = model_cls.load(str(ckpt))
        else:
            print(f"[{model_name}] Create new model")
            model = model_cls(
                n_var=2,
                n_param=2,
                hidden_size=96,
                num_hidden_layers=2,
                dt=DT,
            )

        print(f"[{model_name}] Training...")
        model.fit(
            X,
            y,
            epochs=EPOCHS,
            lr=1e-3,
            batch_size=BATCH_SIZE,
            val_split=VAL_SPLIT,
            checkpoint_dir=str(model_ckpt_dir),
            history_path=str(model_ckpt_dir / "history.json"),
            lr_scheduler=True,
            lr_scheduler_patience=8,
            lr_scheduler_factor=0.3,
            val_every=1,
        )

        final_path = model_ckpt_dir / "model.ckpt"
        model.save(str(final_path))
        print(f"[{model_name}] Saved to {final_path}")

    config = {
        "seed": SEED,
        "dt": DT,
        "variables_ranges": VARIABLES_RANGES,
        "parameters_ranges_train": PARAMETERS_RANGES_TRAIN,
        "num_of_traj": NUM_OF_TRAJ,
        "num_in_traj": NUM_IN_TRAJ,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "val_split": VAL_SPLIT,
        "models": list(models.keys()),
    }
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"Saved config to {CONFIG_PATH}")


if __name__ == "__main__":
    main()
