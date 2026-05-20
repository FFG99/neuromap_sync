"""Per-coordinate or shared MLP with configurable input/output normalization."""

import json
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from tqdm import tqdm

from utils.logger import get_logger

from .nm_manuscript import HistoryCallback


class LREpochLogger(Callback):
    """Пишет сводку метрик в лог каждую эпоху."""

    def __init__(self, log_every_n_epochs: int = 1):
        self.log_every_n_epochs = max(1, int(log_every_n_epochs))
        self._log = get_logger(__name__)

    @staticmethod
    def _metric(trainer, *keys: str):
        for key in keys:
            if key in trainer.callback_metrics:
                return float(trainer.callback_metrics[key].detach().cpu())
            if key in trainer.logged_metrics:
                return float(trainer.logged_metrics[key].detach().cpu())
        return None

    @staticmethod
    def _optimizer_lr(trainer) -> Optional[float]:
        opt = trainer.optimizers
        if opt is None:
            return None
        if isinstance(opt, (list, tuple)):
            opt = opt[0]
        return float(opt.param_groups[0]["lr"])

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        epoch = trainer.current_epoch
        if (epoch + 1) % self.log_every_n_epochs != 0:
            return
        parts = [f"epoch={epoch}"]
        train = self._metric(trainer, "train_loss_epoch", "train_loss")
        val = self._metric(trainer, "val_loss")
        if train is not None:
            parts.append(f"train_mae_next={train:.6f}")
        if val is not None:
            parts.append(f"val_mae_next={val:.6f}")
        for key in ("train_mae_delta", "val_mae_delta"):
            v = self._metric(trainer, f"{key}_epoch", key)
            if v is not None:
                parts.append(f"{key}={v:.6f}")
        lr = self._optimizer_lr(trainer)
        if lr is not None:
            parts.append(f"lr={lr:.2e}")
        self._log.info(" | ".join(parts))

NORM_MODES = ("none", "zscore", "minmax", "zscore_input", "minmax_input")
NORM_MODES_WITH_OUTPUT_SCALE = ("zscore", "minmax")


def _build_mlp(in_features, hidden_size, num_hidden_layers, out_features):
    layers = [nn.Linear(in_features, hidden_size), nn.Tanh()]
    for _ in range(max(0, num_hidden_layers - 1)):
        layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(nn.Tanh())
    layers.append(nn.Linear(hidden_size, out_features))
    return nn.Sequential(*layers)


class NeuroMapManuscriptLR(pl.LightningModule):
    """
    MLP on concat(u, p) -> delta u; loss = MAE(u + delta, u_next).

    use_subnets=True: one MLP per coordinate (scalar output each).
    use_subnets=False: single MLP with vector output.

    norm_mode:
      none         — no scaling
      zscore       — z-score u,p; delta = dt * (g * sd + mu_d)
      minmax       — u,p to [-1,1]; delta = dt * (g * scale_d + bias_d)
      zscore_input — z-score u,p only; delta = dt * g
      minmax_input — minmax u,p only; delta = dt * g
    """

    def __init__(
        self,
        n_var,
        n_param,
        hidden_size=100,
        num_hidden_layers=2,
        dt=0.01,
        lr=1e-3,
        use_subnets=True,
        norm_mode="none",
    ):
        super().__init__()
        if norm_mode not in NORM_MODES:
            raise ValueError(f"norm_mode must be one of {NORM_MODES}, got {norm_mode!r}")

        self.save_hyperparameters(
            "n_var",
            "n_param",
            "hidden_size",
            "num_hidden_layers",
            "dt",
            "lr",
            "use_subnets",
            "norm_mode",
        )

        self.n_var = n_var
        self.n_param = n_param
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.dt = dt
        self.lr = lr
        self.use_subnets = use_subnets
        self.norm_mode = norm_mode

        self.logger_py = get_logger(__name__)
        self._register_norm_buffers()

        in_features = n_var + n_param
        if use_subnets:
            self.subnets = nn.ModuleList(
                [
                    _build_mlp(in_features, hidden_size, num_hidden_layers, 1)
                    for _ in range(n_var)
                ]
            )
            self.network = None
        else:
            self.subnets = None
            self.network = _build_mlp(in_features, hidden_size, num_hidden_layers, n_var)

    def _register_norm_buffers(self):
        n_var, n_param = self.n_var, self.n_param
        mode = self.norm_mode

        if mode in ("zscore", "zscore_input"):
            self.register_buffer("mu_u", torch.zeros(1, n_var))
            self.register_buffer("su", torch.ones(1, n_var))
            self.register_buffer("mu_p", torch.zeros(1, n_param))
            self.register_buffer("sp", torch.ones(1, n_param))
        if mode == "zscore":
            self.register_buffer("mu_d", torch.zeros(1, n_var))
            self.register_buffer("sd", torch.ones(1, n_var))
        if mode in ("minmax", "minmax_input"):
            self.register_buffer("min_u", torch.zeros(1, n_var))
            self.register_buffer("max_u", torch.ones(1, n_var))
            self.register_buffer("min_p", torch.zeros(1, n_param))
            self.register_buffer("max_p", torch.ones(1, n_param))
        if mode == "minmax":
            self.register_buffer("min_d", torch.zeros(1, n_var))
            self.register_buffer("max_d", torch.ones(1, n_var))
            self.register_buffer("scale_d", torch.ones(1, n_var))
            self.register_buffer("bias_d", torch.zeros(1, n_var))

    def _normalize_u_p(self, u, p):
        if self.norm_mode == "none":
            return u, p
        if self.norm_mode in ("zscore", "zscore_input"):
            u_n = (u - self.mu_u) / self.su
            p_n = (p - self.mu_p) / self.sp
            return u_n, p_n
        u_n = 2.0 * (u - self.min_u) / (self.max_u - self.min_u + 1e-8) - 1.0
        p_n = 2.0 * (p - self.min_p) / (self.max_p - self.min_p + 1e-8) - 1.0
        return u_n, p_n

    def _g_to_delta(self, g):
        if self.norm_mode in ("none", "zscore_input", "minmax_input"):
            return self.dt * g
        if self.norm_mode == "zscore":
            return self.dt * (g * self.sd + self.mu_d)
        return self.dt * (g * self.scale_d + self.bias_d)

    def _forward_g(self, u, p):
        u_n, p_n = self._normalize_u_p(u, p)
        z = torch.cat([u_n, p_n], dim=1)
        if self.use_subnets:
            outputs = [subnet(z) for subnet in self.subnets]
            return torch.cat(outputs, dim=1)
        return self.network(z)

    def forward(self, u, p):
        return self._g_to_delta(self._forward_g(u, p))

    def _batch_metrics(self, u, p, y_next):
        d_pred = self.forward(u, p)
        d_true = y_next - u
        mae_next = torch.mean(torch.abs(u + d_pred - y_next))
        mae_delta = torch.mean(torch.abs(d_pred - d_true))
        return mae_next, mae_delta

    def training_step(self, batch, batch_idx):
        X_batch, y_batch = batch
        u_batch = X_batch[:, : self.n_var]
        p_batch = X_batch[:, self.n_var : self.n_var + self.n_param]
        mae_next, mae_delta = self._batch_metrics(u_batch, p_batch, y_batch)
        self.log("train_loss", mae_next, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_mae_delta", mae_delta, on_step=False, on_epoch=True, prog_bar=False)
        return mae_next

    def validation_step(self, batch, batch_idx):
        X_batch, y_batch = batch
        u_batch = X_batch[:, : self.n_var]
        p_batch = X_batch[:, self.n_var : self.n_var + self.n_param]
        mae_next, mae_delta = self._batch_metrics(u_batch, p_batch, y_batch)
        self.log("val_loss", mae_next, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_mae_delta", mae_delta, on_step=False, on_epoch=True, prog_bar=False)
        return mae_next

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        if hasattr(self.hparams, "lr_scheduler") and self.hparams.lr_scheduler:
            scheduler_frequency = max(1, int(getattr(self.hparams, "val_every", 1)))
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.hparams.lr_scheduler_factor,
                patience=self.hparams.lr_scheduler_patience,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": scheduler_frequency,
                },
            }
        return optimizer

    def _compute_statistics(self, X, y_next):
        """y_next: u(n+1); statistics for delta use d = y_next - u."""
        u = X[:, : self.n_var]
        p = X[:, self.n_var : self.n_var + self.n_param]
        d = y_next - u

        if np.any(np.std(d, axis=0) < 1e-8):
            self.logger_py.warning("Обнаружены константные приращения (std < 1e-8)")

        def _set_buffer(name, value):
            buf = getattr(self, name)
            buf.copy_(torch.tensor(value, dtype=torch.float32).reshape_as(buf))

        if self.norm_mode in ("zscore", "zscore_input"):
            _set_buffer("mu_u", np.mean(u, axis=0, keepdims=True))
            _set_buffer("su", np.std(u, axis=0, keepdims=True))
            self.su.clamp_(min=1e-6)
            _set_buffer("mu_p", np.mean(p, axis=0, keepdims=True))
            _set_buffer("sp", np.std(p, axis=0, keepdims=True))
            self.sp.clamp_(min=1e-6)

        if self.norm_mode == "zscore":
            _set_buffer("mu_d", np.mean(d, axis=0, keepdims=True))
            _set_buffer("sd", np.std(d, axis=0, keepdims=True))
            self.sd.clamp_(min=1e-6)

        if self.norm_mode in ("minmax", "minmax_input"):
            _set_buffer("min_u", np.min(u, axis=0, keepdims=True))
            _set_buffer("max_u", np.max(u, axis=0, keepdims=True))
            _set_buffer("min_p", np.min(p, axis=0, keepdims=True))
            _set_buffer("max_p", np.max(p, axis=0, keepdims=True))

        if self.norm_mode == "minmax":
            d_min = np.min(d, axis=0, keepdims=True)
            d_max = np.max(d, axis=0, keepdims=True)
            _set_buffer("min_d", d_min)
            _set_buffer("max_d", d_max)
            _set_buffer("scale_d", (d_max - d_min) / 2.0)
            self.scale_d.clamp_(min=1e-6)
            _set_buffer("bias_d", (d_max + d_min) / 2.0)

        self.logger_py.info(
            "Статистики нормализации (%s): |Δu| mean=%.4e std=%.4e",
            self.norm_mode,
            float(np.mean(np.abs(d))),
            float(np.std(d)),
        )

    def fit(
        self,
        X,
        y,
        epochs=1000,
        lr=1e-3,
        batch_size=64,
        val_split=0.1,
        val_every=1,
        log_every=100,
        verbose=True,
        num_workers=0,
        checkpoint_dir=None,
        history_path=None,
        ckpt_path=None,
        gradient_clip_val=None,
        gradient_clip_algorithm="norm",
        early_stopping_patience=None,
        lr_scheduler=False,
        lr_scheduler_patience=10,
        lr_scheduler_factor=0.1,
    ):
        """y must be next state u(n+1), not delta."""
        self.lr = lr
        self.hparams.val_every = int(val_every)
        if lr_scheduler:
            self.hparams.lr_scheduler = True
            self.hparams.lr_scheduler_patience = lr_scheduler_patience
            self.hparams.lr_scheduler_factor = lr_scheduler_factor

        if ckpt_path is None and checkpoint_dir is not None:
            checkpoint_dir_path = Path(checkpoint_dir)
            if checkpoint_dir_path.exists():
                checkpoint_files = list(checkpoint_dir_path.glob("epoch=*.ckpt"))
                if checkpoint_files:
                    ckpt_path = str(max(checkpoint_files, key=lambda p: p.stat().st_mtime))
                    self.logger_py.info(f"Найден чекпоинт для продолжения: {ckpt_path}")

        N = len(X)
        n_train = int(N * (1 - val_split))
        X_train, X_val = X[:n_train], X[n_train:]
        y_train, y_val = y[:n_train], y[n_train:]

        d_train = y_train - X_train[:, : self.n_var]
        self.logger_py.info(
            "Датасет |Δu|: mean=%.4e std=%.4e max=%.4e",
            float(np.mean(np.abs(d_train))),
            float(np.std(d_train)),
            float(np.max(np.abs(d_train))),
        )

        if self.norm_mode != "none":
            self._compute_statistics(X_train, y_train)
        else:
            self.logger_py.info("Нормализация отключена (norm_mode=none)")

        self.logger_py.info(
            "Старт обучения: train=%s | val=%s | epochs=%s | batch=%s | lr=%s | "
            "use_subnets=%s | norm_mode=%s | hidden=%s×%s",
            f"{len(X_train):,}",
            f"{len(X_val):,}",
            epochs,
            batch_size,
            lr,
            self.use_subnets,
            self.norm_mode,
            self.hidden_size,
            self.num_hidden_layers,
        )

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        val_dataloader = None
        if val_split > 0:
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
            val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
            )

        history_callback = HistoryCallback()
        callbacks = [
            history_callback,
            LREpochLogger(log_every_n_epochs=1),
            TQDMProgressBar(refresh_rate=1 if verbose else 0),
        ]
        if lr_scheduler:
            callbacks.append(LearningRateMonitor(logging_interval="epoch"))

        checkpoint_callback = None
        if val_split > 0:
            checkpoint_kwargs = {
                "monitor": "val_loss",
                "mode": "min",
                "save_top_k": 3,
                "verbose": verbose,
                "save_last": True,
            }
            if checkpoint_dir is not None:
                checkpoint_kwargs["dirpath"] = checkpoint_dir
            checkpoint_callback = ModelCheckpoint(**checkpoint_kwargs)
            callbacks.append(checkpoint_callback)

        if early_stopping_patience is not None and val_split > 0:
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss",
                    patience=early_stopping_patience,
                    mode="min",
                    verbose=verbose,
                )
            )

        log_every_n_steps = log_every if log_every > 0 else 50

        pl_loggers = True
        if checkpoint_dir is not None:
            log_dir = Path(checkpoint_dir) / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            pl_loggers = [
                TensorBoardLogger(save_dir=str(log_dir), name="tb"),
                CSVLogger(save_dir=str(log_dir), name="csv"),
            ]
            self.logger_py.info("Логи Lightning: %s (TensorBoard + CSV)", log_dir)

        trainer = pl.Trainer(
            max_epochs=epochs,
            callbacks=callbacks,
            enable_progress_bar=verbose,
            log_every_n_steps=log_every_n_steps,
            check_val_every_n_epoch=val_every if val_split > 0 else 1,
            logger=pl_loggers,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
            enable_checkpointing=checkpoint_dir is not None,
        )
        trainer.fit(self, train_dataloader, val_dataloader, ckpt_path=ckpt_path)

        if history_path is not None:
            history = history_callback.get_history()
            history_path = Path(history_path)
            history_path.parent.mkdir(parents=True, exist_ok=True)
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
            self.logger_py.info(f"История обучения сохранена в {history_path}")

        self.training_history = history_callback.get_history()

        if checkpoint_callback is not None and checkpoint_callback.best_model_score is not None:
            self.logger_py.info(
                f"Готово. Лучшая val_mae_next_state: {checkpoint_callback.best_model_score:.6f} "
                f"(эпоха {checkpoint_callback.best_model_path.split('epoch=')[-1].split('-')[0]})"
            )
            if checkpoint_callback.best_model_path:
                self.logger_py.info(f"Загружаем лучшую модель из {checkpoint_callback.best_model_path}")
                checkpoint = torch.load(checkpoint_callback.best_model_path, map_location=self.device)
                state_dict = checkpoint["state_dict"]
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    if k.startswith("model."):
                        new_state_dict[k[6:]] = v
                    else:
                        new_state_dict[k] = v
                self.load_state_dict(new_state_dict, strict=True)
        else:
            self.logger_py.info("Готово.")

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
            u = X_tensor[:, : self.n_var]
            p = X_tensor[:, self.n_var : self.n_var + self.n_param]
            return self.forward(u, p).cpu().numpy()

    def compute_d_and_jacobian(self, u, p):
        self.eval()
        u = np.asarray(u, dtype=np.float64).ravel()
        p = np.asarray(p, dtype=np.float64).ravel()
        with torch.enable_grad():
            u_t = torch.tensor(u, dtype=torch.float32, device=self.device).reshape(1, -1).requires_grad_(True)
            p_t = torch.tensor(p, dtype=torch.float32, device=self.device).reshape(1, -1)
            d_vec = self.forward(u_t, p_t).squeeze(0)
            j_rows = []
            for i in range(self.n_var):
                gr = torch.autograd.grad(d_vec[i], u_t, retain_graph=True)[0]
                j_rows.append(gr.squeeze(0))
            j_d = torch.stack(j_rows, dim=0)
        return d_vec.detach().cpu().numpy(), j_d.detach().cpu().numpy()

    def find_fixed_point(self, p, u0, tol=1e-10, method="hybr"):
        from scipy.optimize import root

        p = np.atleast_1d(p).astype(np.float64)
        u0 = np.atleast_1d(u0).astype(np.float64)

        def residual(u):
            d, _ = self.compute_d_and_jacobian(u, p)
            return d

        def jacobian(u):
            _, j_d = self.compute_d_and_jacobian(u, p)
            return j_d

        result = root(residual, u0, method=method, jac=jacobian, tol=tol)
        return result.x, result.success, result

    def find_all_fixed_points(
        self, p, bounds, n_grid=10, tol=1e-10, unique_tol=1e-6, method="hybr"
    ):
        grids = [np.linspace(b[0], b[1], n_grid) for b in bounds]
        mesh = np.meshgrid(*grids, indexing="ij")
        initial_guesses = np.stack([m.ravel() for m in mesh], axis=1)
        found_points = []
        for u0 in initial_guesses:
            u_star, converged, _ = self.find_fixed_point(p, u0, tol=tol, method=method)
            if not converged:
                continue
            in_bounds = all(
                bounds[i][0] - unique_tol <= u_star[i] <= bounds[i][1] + unique_tol
                for i in range(len(bounds))
            )
            if not in_bounds:
                continue
            is_unique = True
            for existing in found_points:
                if np.linalg.norm(u_star - existing) < unique_tol:
                    is_unique = False
                    break
            if is_unique:
                found_points.append(u_star)
        multipliers = [self.compute_fixed_point_multipliers(fp, p) for fp in found_points]
        return found_points, multipliers

    def compute_fixed_point_multipliers(self, u_star, p):
        _, j_d = self.compute_d_and_jacobian(u_star, p)
        j_n = np.eye(self.n_var) + j_d
        return np.linalg.eigvals(j_n)

    def simulate(self, u0, p, n_steps, verbose=True, divergence_threshold=1e5):
        u0 = np.atleast_2d(u0)
        p = np.atleast_2d(p)
        trajectory = [u0.copy()]
        u_current = u0.copy()
        iterator = range(n_steps)
        if verbose:
            iterator = tqdm(iterator, desc="Симуляция", unit="шаг", ncols=100)
        for _ in iterator:
            X_step = np.concatenate([u_current, p], axis=1)
            d = self.predict(X_step)
            u_current = u_current + d
            if np.linalg.norm(u_current) > divergence_threshold:
                self.logger_py.info(
                    f"Траектория разошлась (норма = {np.linalg.norm(u_current):.2e} > {divergence_threshold:.2e})"
                )
                return None
            trajectory.append(u_current.copy())
        return np.concatenate(trajectory, axis=0)

    @classmethod
    def load(cls, path, device=None):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Файл не найден: {path.absolute()}")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = torch.load(str(path), map_location=device, weights_only=False)
        if "pytorch-lightning_version" in checkpoint:
            return cls.load_from_checkpoint(str(path), map_location=device)
        if "_format" in checkpoint and checkpoint["_format"] in (
            "neuromap_lr_v2.0",
            "neuromap_lr_v1.0",
        ):
            hparams = dict(checkpoint.get("hyper_parameters", {}))
            required = ["n_var", "n_param", "hidden_size", "dt", "lr"]
            missing = [x for x in required if x not in hparams]
            if missing:
                raise ValueError(f"Отсутствуют обязательные гиперпараметры: {missing}")
            hparams.setdefault("num_hidden_layers", 2)
            hparams.setdefault("use_subnets", True)
            hparams.setdefault("norm_mode", "none")
            if checkpoint["_format"] == "neuromap_lr_v1.0":
                hparams.pop("target_mode", None)
            ctor_keys = (
                "n_var",
                "n_param",
                "hidden_size",
                "num_hidden_layers",
                "dt",
                "lr",
                "use_subnets",
                "norm_mode",
            )
            model = cls(**{k: hparams[k] for k in ctor_keys if k in hparams})
            model.load_state_dict(checkpoint["state_dict"], strict=False)
            model.to(device)
            if not hasattr(model, "training_history") or model.training_history is None:
                history_path = path.parent / f"{path.stem}_history.json"
                if history_path.exists():
                    try:
                        with open(history_path, "r", encoding="utf-8") as f:
                            model.training_history = json.load(f)
                    except Exception as e:
                        model.logger_py.warning(f"Не удалось загрузить историю из {history_path}: {e}")
            return model
        raise ValueError(f"Неподдерживаемый формат чекпоинта: {path}")

    def save(self, path, save_history=True):
        path = Path(path).with_suffix(".ckpt")
        path.parent.mkdir(parents=True, exist_ok=True)
        hparams = {
            "n_var": int(self.n_var),
            "n_param": int(self.n_param),
            "hidden_size": int(self.hidden_size),
            "num_hidden_layers": int(self.num_hidden_layers),
            "dt": float(self.dt),
            "lr": float(self.lr),
            "use_subnets": bool(self.use_subnets),
            "norm_mode": self.norm_mode,
        }
        checkpoint = {
            "state_dict": self.state_dict(),
            "hyper_parameters": hparams,
            "_format": "neuromap_lr_v2.0",
            "torch_version": torch.__version__,
            "saved_at": datetime.now().isoformat(),
        }
        torch.save(checkpoint, str(path))
        if save_history and hasattr(self, "training_history"):
            history_path = path.parent / f"{path.stem}_history.json"
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(self.training_history, f, indent=2)
        return path


class NeuroMapManuscriptLRShared(NeuroMapManuscriptLR):
    """Single shared MLP for all coordinates (use_subnets=False)."""

    def __init__(self, *args, **kwargs):
        kwargs["use_subnets"] = False
        super().__init__(*args, **kwargs)
