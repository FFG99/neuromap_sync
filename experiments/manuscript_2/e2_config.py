"""
Общие настройки эксперимента manuscript_2 (vdp_mod2: параметры λ, μ).

Диапазоны фазового пространства и шаг интегрирования согласованы с manuscript_1 по масштабу;
подбор интервалов по λ и μ под задачу можно менять здесь.
"""

SEED = 42

# x₁, x₂ — одинаковы для обоих прогонов
VARIABLES_RANGES = [(-3.3, 3.3), (-29.0, 29.0)]

# λ , μ — общая «большая» область в параметрическом пространстве (одинакова в обоих экспериментах)
PARAMETERS_RANGES = [(-3.0, 1.0), (0.02, 0.1)]

EXCLUDE_VARIABLES_RANGES = [(-2.24, 2.24), (-12.7, 12.7)]

NUM_OF_TRAJ = 12_500
NUM_IN_TRAJ = 10
DT = 0.01

# Архитектура как в manuscript_1 / e1_train
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
