# manuscript_lr

Тот же протокол, что `experiments/manuscript_1` (vdp_mod1, 2 скрытых слоя), но:

- модель `NeuroMapManuscriptLR` (loss = MAE next state);
- **125_000** траекторий (×10);
- **4 варианта** в одном прогоне: подсети / общая сеть × Z-score / без нормализации.

| Вариант | `use_subnets` | `norm_mode` | Чекпоинт |
|---------|---------------|-------------|----------|
| `subnets_none` | да | `none` | `checkpoints/subnets_none/model.ckpt` |
| `subnets_zscore` | да | `zscore` | `checkpoints/subnets_zscore/model.ckpt` |
| `shared_none` | нет | `none` | `checkpoints/shared_none/model.ckpt` |
| `shared_zscore` | нет | `zscore` | `checkpoints/shared_zscore/model.ckpt` |

## Обучение

```bash
# все 4 модели (датасет кэшируется в datasets/e1_vdp_mod1.npz)
python experiments/manuscript_lr/e1_train.py

# один вариант
python experiments/manuscript_lr/e1_train.py --variant subnets_zscore

# пропустить уже обученные
python experiments/manuscript_lr/e1_train.py --skip-if-done
```

Логи: `checkpoints/<variant>/logs/` (TensorBoard + CSV), `history.json`.

```bash
tensorboard --logdir experiments/manuscript_lr/checkpoints
```

## Тесты (precompute)

**P(НТ) для всех 4 моделей** (из корня репозитория):

```bash
# 1) первый вариант: ODE + neuromap
python experiments/manuscript_lr/e1_precompute_fixed_point_probability.py --variant subnets_none

# 2) остальные три: только neuromap (ODE тот же)
for v in subnets_zscore shared_none shared_zscore; do
  python experiments/manuscript_lr/e1_precompute_fixed_point_probability.py --variant "$v" --skip-ode
done
```

Результаты в `experiments/manuscript_lr/results/<variant>/`:
`e1_ode_fixed_point_probability.npz`, `e1_neuromap_fixed_point_probability.npz`.

Один вариант:

```bash
python experiments/manuscript_lr/e1_precompute_fixed_point_probability.py --variant subnets_zscore
```

Все 4 варианта × 3 скрипта (P(НТ), басейны, скан):

```bash
python experiments/manuscript_lr/e1_precompute_all.py
```

## Визуализация

```bash
jupyter notebook experiments/manuscript_lr/e1_analysis.ipynb
```

Ноутбук загружает все 4 модели, сравнивает кривые обучения, траектории и (при наличии npz) P(НТ), басейны, скан НТ.
