# Manuscript-0303 Exact Stages

Поэтапный репрод с архитектурой `NeuroMapManuscriptSubnets`.

## Model (3): vdp_mod1, параметры `(lambda, beta)`

1) Генерация train/val датасетов:

```bash
python experiments/manuscript_0303_exact/01_generate_dataset_model3.py
```

2) Обучение Subnets-модели:

```bash
python experiments/manuscript_0303_exact/02_train_model3_subnets.py
```

3) Предвычисление артефактов статьи:

```bash
python experiments/manuscript_0303_exact/03_precompute_model3.py
```

## Model (4): vdp_mod2, параметры `(lambda, mu)`

1) Генерация train/val датасетов:

```bash
python experiments/manuscript_0303_exact/11_generate_dataset_model4.py
```

2) Обучение Subnets-модели:

```bash
python experiments/manuscript_0303_exact/12_train_model4_subnets.py
```

3) Предвычисление артефактов статьи:

```bash
python experiments/manuscript_0303_exact/13_precompute_model4.py
```

Все артефакты сохраняются в `experiments/manuscript_0303_exact/artifacts/`.
