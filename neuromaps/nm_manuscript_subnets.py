# -*- coding: utf-8 -*-
"""NeuroMap manuscript variant with one subnet per output variable."""

import json
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from tqdm import tqdm

from utils.logger import get_logger

from .nm_manuscript import HistoryCallback


class NeuroMapManuscriptSubnets(pl.LightningModule):
    """
    Manuscript-style decomposition: one trainable subnet per output coordinate.

    Each subnet maps [u, p] -> scalar g_i, and d_i = dt * g_i.
    """

    def __init__(self, n_var, n_param, hidden_size=100, num_hidden_layers=2, dt=0.01, lr=1e-3):
        super().__init__()
        self.save_hyperparameters(
            "n_var", "n_param", "hidden_size", "num_hidden_layers", "dt", "lr"
        )

        self.n_var = n_var
        self.n_param = n_param
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.dt = dt
        self.lr = lr

        self.logger_py = get_logger(__name__)
        self.activation = nn.Tanh()

        self.subnets = nn.ModuleList([self._build_subnet() for _ in range(n_var)])
        self.criterion = nn.L1Loss()

    def _build_subnet(self):
        in_features = self.n_var + self.n_param
        layers = [nn.Linear(in_features, self.hidden_size), self.activation]
        for _ in range(max(0, self.num_hidden_layers - 1)):
            layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            layers.append(self.activation)
        layers.append(nn.Linear(self.hidden_size, 1))
        return nn.Sequential(*layers)

    def forward(self, u, p):
        z = torch.cat([u, p], dim=1)
        outputs = [subnet(z) for subnet in self.subnets]
        g = torch.cat(outputs, dim=1)
        return self.dt * g

    def training_step(self, batch, batch_idx):
        X_batch, y_batch = batch
        u_batch = X_batch[:, : self.n_var]
        p_batch = X_batch[:, self.n_var : self.n_var + self.n_param]
        pred = self.forward(u_batch, p_batch)
        loss = self.criterion(pred, y_batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X_batch, y_batch = batch
        u_batch = X_batch[:, : self.n_var]
        p_batch = X_batch[:, self.n_var : self.n_var + self.n_param]
        pred = self.forward(u_batch, p_batch)
        loss = self.criterion(pred, y_batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        if hasattr(self.hparams, "lr_scheduler") and self.hparams.lr_scheduler:
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
                    "frequency": 1,
                },
            }
        return optimizer

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
        self.lr = lr
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

        self.logger_py.info(f"Тренировка: {len(X_train):,} | Валидация: {len(X_val):,}")

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
        callbacks = [history_callback, TQDMProgressBar(refresh_rate=1 if verbose else 0)]
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
        trainer = pl.Trainer(
            max_epochs=epochs,
            callbacks=callbacks,
            enable_progress_bar=verbose,
            log_every_n_steps=log_every_n_steps,
            check_val_every_n_epoch=val_every if val_split > 0 else 1,
            logger=True,
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
                f"Готово. Лучшая валидация: {checkpoint_callback.best_model_score:.6f} "
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

    def compute_d_and_jacobian(self, u, p):
        """Compute increment d and Jacobian dd/du via autograd."""
        self.eval()
        u = np.asarray(u, dtype=np.float64).ravel()
        p = np.asarray(p, dtype=np.float64).ravel()
        with torch.enable_grad():
            u_t = torch.tensor(u, dtype=torch.float32, device=self.device).reshape(1, -1).requires_grad_(True)
            p_t = torch.tensor(p, dtype=torch.float32, device=self.device).reshape(1, -1)
            d_vec = self.forward(u_t, p_t).squeeze(0)
            J_rows = []
            for i in range(self.n_var):
                gr = torch.autograd.grad(d_vec[i], u_t, retain_graph=True)[0]
                J_rows.append(gr.squeeze(0))
            J_d = torch.stack(J_rows, dim=0)
        d_np = d_vec.detach().cpu().numpy()
        return d_np, J_d.detach().cpu().numpy()

    def find_fixed_point(self, p, u0, tol=1e-10, method="hybr"):
        from scipy.optimize import root

        p = np.atleast_1d(p).astype(np.float64)
        u0 = np.atleast_1d(u0).astype(np.float64)

        def residual(u):
            d, _ = self.compute_d_and_jacobian(u, p)
            return d

        def jacobian(u):
            _, J_d = self.compute_d_and_jacobian(u, p)
            return J_d

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
        _, J_d = self.compute_d_and_jacobian(u_star, p)
        J_n = np.eye(self.n_var) + J_d
        return np.linalg.eigvals(J_n)

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
        if "_format" in checkpoint and checkpoint["_format"] == "neuromap_subnets_v1.0":
            hparams = checkpoint.get("hyper_parameters", {})
            required = ["n_var", "n_param", "hidden_size", "num_hidden_layers", "dt", "lr"]
            missing = [x for x in required if x not in hparams]
            if missing:
                raise ValueError(f"Отсутствуют обязательные гиперпараметры: {missing}")
            model = cls(**hparams)
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
        }
        checkpoint = {
            "state_dict": self.state_dict(),
            "hyper_parameters": hparams,
            "_format": "neuromap_subnets_v1.0",
            "torch_version": torch.__version__,
            "saved_at": datetime.now().isoformat(),
        }
        torch.save(checkpoint, str(path))
        if save_history and hasattr(self, "training_history"):
            history_path = path.parent / f"{path.stem}_history.json"
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(self.training_history, f, indent=2)
        return path
