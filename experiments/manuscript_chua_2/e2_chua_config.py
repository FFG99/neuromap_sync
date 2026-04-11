"""
Общие настройки эксперимента manuscript_chua_2 (система Чуа, 3×5).

Стиль как у manuscript_2: те же гиперпараметры сети по умолчанию; фазовый бокс и
параметры согласованы с experiments/long_trajs_chua (фиксированные параметры —
интервалы вида (a, a)).

Порог REJECT_AMPLITUDE_ABOVE подбирается в amplitude_threshold_sweep.ipynb
(амплитуда = ||ptp||_2 по точкам сечения Пуанкаре S = y, как в calculate_dynamic_regime).
"""

SEED = 42

VARIABLES_RANGES = [(-15.0, 15.0), (-4.0, 4.0), (-18.0, 18.0)]

PARAMETERS_RANGES = [
    (8.4, 8.4),
    (12.0, 12.0),
    (0.0, 0.0),
    (-0.12, -0.12),
    (-1.15, -1.15),
]

NUM_OF_TRAJ = 12_500
NUM_IN_TRAJ = 10
DT = 0.01

HIDDEN_SIZE = 100
NUM_HIDDEN_LAYERS = 2
EPOCHS = 1000
LR = 1e-3
BATCH_SIZE = 256
VAL_SPLIT = 0.2
LR_SCHEDULER = True
LR_SCHEDULER_PATIENCE = 10
LR_SCHEDULER_FACTOR = 0.1
VAL_EVERY = 1

N_TRANSIENT = 200

DATASET_EXCLUDE_LARGE_NPZ = "datasets/chua_exclude_large_lc.npz"

REJECT_AMPLITUDE_ABOVE = 3
