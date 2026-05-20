"""Общий конфиг manuscript_lr (vdp_mod1, ×10 траекторий к manuscript_1)."""

from pathlib import Path

SEED = 42
DT = 0.01

VARIABLES_RANGES = [(-10.19, 10.18), (-136.5, 136.5)]
PARAMETERS_RANGES = [(-3.0, 1.0), (0.02, 0.1)]

NUM_OF_TRAJ = 125_000
NUM_IN_TRAJ = 10

N_VAR = len(VARIABLES_RANGES)
N_PARAM = len(PARAMETERS_RANGES)

HIDDEN_SIZE = 100
NUM_HIDDEN_LAYERS = 2

EPOCHS = 1000
BATCH_SIZE = 512
LR = 1e-3
VAL_SPLIT = 0.2
LR_SCHEDULER_PATIENCE = 10
LR_SCHEDULER_FACTOR = 0.9

# Пути относительно этого файла (работает из любого cwd)
EXP_DIR = Path(__file__).resolve().parent
CHECKPOINTS_ROOT = EXP_DIR / "checkpoints"
RESULTS_DIR = EXP_DIR / "results"
DATASET_PATH = EXP_DIR / "datasets" / "e1_vdp_mod1.npz"

# 2×2: подсети / общая сеть × Z-score / без нормализации
TRAINING_VARIANTS = (
    {"name": "subnets_none", "use_subnets": True, "norm_mode": "none"},
    {"name": "subnets_zscore", "use_subnets": True, "norm_mode": "zscore"},
    {"name": "shared_none", "use_subnets": False, "norm_mode": "none"},
    {"name": "shared_zscore", "use_subnets": False, "norm_mode": "zscore"},
)

VARIANT_NAMES = tuple(v["name"] for v in TRAINING_VARIANTS)


def checkpoint_dir(variant_name: str) -> Path:
    return CHECKPOINTS_ROOT / variant_name


def model_ckpt_path(variant_name: str) -> Path:
    return checkpoint_dir(variant_name) / "model.ckpt"


def variant_by_name(name: str) -> dict:
    for v in TRAINING_VARIANTS:
        if v["name"] == name:
            return v
    raise KeyError(f"Неизвестный вариант {name!r}. Доступны: {VARIANT_NAMES}")


def results_dir(variant_name: str) -> Path:
    return RESULTS_DIR / variant_name


DEFAULT_VARIANT = "subnets_zscore"
