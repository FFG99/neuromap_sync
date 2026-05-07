# Manuscript-0303 Exact Stages

Поэтапный репрод с архитектурой `NeuroMapManuscriptEq8`.
Train-этапы запускаются в строгом paper-режиме:
- `Nh=100`,
- структура Eq.(8): отдельные параметры по координатам (`A_i, B_i, b_i, a_i, c_i, d_i, gamma_i`),
- `Adam`, `lr=1e-3`,
- с `ReduceLROnPlateau` (`patience=10`, `factor=0.1`),
- текущий `lr` логгируется каждый epoch в метрику `lr`.

## Model (3): vdp_mod1, параметры `(lambda, beta)`

1) Генерация train/val датасетов:

```bash
python experiments/manuscript_0303_exact/01_generate_dataset_model3.py
```

2) Обучение Eq8-модели:

```bash
python experiments/manuscript_0303_exact/02_train_model3_subnets.py
```

Перед повторным «чистым» запуском удалите старые чекпоинты:
`experiments/manuscript_0303_exact/artifacts/model3/checkpoints_eq8/`

3) Предвычисление артефактов статьи:

```bash
python experiments/manuscript_0303_exact/03_precompute_model3.py
```

## Model (4): vdp_mod2, параметры `(lambda, mu)`

1) Генерация train/val датасетов:

```bash
python experiments/manuscript_0303_exact/11_generate_dataset_model4.py
```

2) Обучение Eq8-модели:

```bash
python experiments/manuscript_0303_exact/12_train_model4_subnets.py
```

Перед повторным «чистым» запуском удалите старые чекпоинты:
`experiments/manuscript_0303_exact/artifacts/model4/checkpoints_eq8/`

3) Предвычисление артефактов статьи:

```bash
python experiments/manuscript_0303_exact/13_precompute_model4.py
```

Все артефакты сохраняются в `experiments/manuscript_0303_exact/artifacts/`.
