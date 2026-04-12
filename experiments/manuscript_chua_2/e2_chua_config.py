"""
Общие настройки эксперимента manuscript_chua_2 (система Чуа, 3×5).

Стиль как у manuscript_2: те же гиперпараметры сети по умолчанию; фазовый бокс и
параметры согласованы с experiments/long_trajs_chua (фиксированные параметры —
интервалы вида (a, a)).

Порог REJECT_AMPLITUDE_ABOVE подбирается в amplitude_threshold_sweep.ipynb.
Амплитуда: ||ptp(U)||_2 по окну полной RK-траектории (см. full_trajectory_ptp_norm:
AMPLITUDE_BURN_RK_STEPS + AMPLITUDE_RECORD_RK_STEPS шагов).
"""

SEED = 42

VARIABLES_RANGES = [(-15.0, 15.0), (-4.0, 4.0), (-18.0, 18.0)]

PARAMETERS_RANGES = [
    (8.41, 8.41),
    (12.23, 12.23),
    (0.0435, 0.0435),
    (-1.366, -1.366),
    (-0.17, -0.17)
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

AMPLITUDE_BURN_RK_STEPS = 2000
AMPLITUDE_RECORD_RK_STEPS = 4000

RK_DIVERGENCE_THRESHOLD = 1e5

DATASET_EXCLUDE_LARGE_NPZ = "datasets/chua_exclude_large_lc.npz"

# Масштаб — по полной RK-траектории
REJECT_AMPLITUDE_ABOVE = 60
