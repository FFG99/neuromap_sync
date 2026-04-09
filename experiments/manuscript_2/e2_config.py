"""
Общие настройки эксперимента manuscript_2 (vdp_mod2: параметры λ, μ).

Диапазоны фазового пространства и шаг интегрирования согласованы с manuscript_1 по масштабу;
подбор интервалов по λ и μ под задачу можно менять здесь.
"""

SEED = 42

# x₁, x₂ — одинаковы для обоих прогонов
VARIABLES_RANGES = [(-10.19, 10.18), (-136.5, 136.5)]

# λ , μ — общая «большая» область в параметрическом пространстве (одинакова в обоих экспериментах)
PARAMETERS_RANGES = [(-3.0, 1.0), (0.02, 0.12)]

# Второй прогон: та же PARAMETERS_RANGES, но вырезаем подпрямоугольник.
# По λ берём весь интервал — тогда отбор отбрасывает только слой по μ (см. докстринг generate_pairs_dataset_finite).
EXCLUDE_PARAMETERS_RANGES = [(-3.0, 1.0), (0.055, 0.085)]

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
