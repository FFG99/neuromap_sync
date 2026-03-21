# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    TQDMProgressBar,
    EarlyStopping,
    LearningRateMonitor
)
from pytorch_lightning.callbacks import Callback

import json
from datetime import datetime
from utils.logger import get_logger


class HistoryCallback(Callback):
    """Callback для сбора истории обучения"""

    def __init__(self):
        super().__init__()
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'epoch': [],
            'best_val_loss': float('inf'),
            'best_epoch': -1
        }
        self.last_val_loss = None

    def on_train_epoch_end(self, trainer, pl_module):
        """Собираем метрики в конце каждой эпохи обучения"""
        epoch = trainer.current_epoch

        metrics = trainer.callback_metrics
        logged_metrics = trainer.logged_metrics

        train_loss = None
        for key in ['train_loss_epoch', 'train_loss']:
            if key in metrics:
                train_loss = float(metrics[key].cpu())
                break
            elif key in logged_metrics:
                train_loss = float(logged_metrics[key].cpu())
                break

        if train_loss is not None:
            self.history['train_loss'].append(train_loss)
        elif len(self.history['train_loss']) > 0:
            self.history['train_loss'].append(self.history['train_loss'][-1])

        val_loss = None
        for key in ['val_loss']:
            if key in metrics:
                val_loss = float(metrics[key].cpu())
                self.last_val_loss = val_loss
                break
            elif key in logged_metrics:
                val_loss = float(logged_metrics[key].cpu())
                self.last_val_loss = val_loss
                break

        if val_loss is not None:
            self.history['val_loss'].append(val_loss)

            # Отслеживаем лучшую модель
            if val_loss < self.history['best_val_loss']:
                self.history['best_val_loss'] = val_loss
                self.history['best_epoch'] = epoch
        elif self.last_val_loss is not None:
            self.history['val_loss'].append(self.last_val_loss)

        self.history['epoch'].append(epoch)

    def get_history(self):
        """Возвращает историю обучения"""
        return self.history.copy()


class NeuroMapManuscript(pl.LightningModule):
    """NeuroMapOriginal implements the neural network mapping described in
    *“Neural network maps for two-parameter modeling of bistability and codimension-two
    bifurcations in two-dimensional flow dynamical systems”*.
    The network learns a mapping from the current state `u` and parameters `p`
    to the increment `d = u_{n+1} - u_n`.  The output of the network is
    scaled by the integration step `dt`; no additional scaling with respect to
    the statistics of `u` is applied (the original article does not use
    `mu_u` and `su` for denormalising the output)."""

    def __init__(self, n_var, n_param,
                 hidden_size=50, num_hidden_layers=1,
                 dt=0.01, lr=1e-3):
        super().__init__()
        self.save_hyperparameters(
            'n_var', 'n_param', 'hidden_size',
            'num_hidden_layers', 'dt', 'lr'
        )

        self.n_var = n_var
        self.n_param = n_param
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.dt = dt
        self.lr = lr

        self.logger_py = get_logger(__name__)

        self.activation = nn.Tanh()
        if num_hidden_layers == 1:
            self.hidden = nn.Linear(n_var + n_param, hidden_size)
            self.output = nn.Linear(hidden_size, n_var)
            self.hidden_layers = None
        else:
            self.hidden = None
            layers = [nn.Linear(n_var + n_param, hidden_size)]
            for _ in range(num_hidden_layers - 1):
                layers.append(nn.Linear(hidden_size, hidden_size))
            self.hidden_layers = nn.ModuleList(layers)
            self.output = nn.Linear(hidden_size, n_var)

        self.criterion = nn.L1Loss()

    def forward(self, u, p):
        """Прямой проход – возвращает приращение `d`."""
        z = torch.cat([u, p], dim=1)
        if self.num_hidden_layers == 1:
            h = self.activation(self.hidden(z))
        else:
            h = z
            for layer in self.hidden_layers:
                h = self.activation(layer(h))
        g = self.output(h)

        # The network output `g` is directly the increment scaled by dt.
        d = self.dt * g
        return d

    def training_step(self, batch, batch_idx):
        X_batch, y_batch = batch
        u_batch = X_batch[:, :self.n_var]
        p_batch = X_batch[:, self.n_var:self.n_var + self.n_param]

        pred = self.forward(u_batch, p_batch)
        loss = self.criterion(pred, y_batch)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X_batch, y_batch = batch
        u_batch = X_batch[:, :self.n_var]
        p_batch = X_batch[:, self.n_var:self.n_var + self.n_param]

        pred = self.forward(u_batch, p_batch)
        loss = self.criterion(pred, y_batch)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        # Добавляем scheduler если указан в гиперпараметрах
        if hasattr(self.hparams, 'lr_scheduler') and self.hparams.lr_scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.hparams.lr_scheduler_factor,
                patience=self.hparams.lr_scheduler_patience
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss',
                    'interval': 'epoch',
                    'frequency': 1
                }
            }
        return optimizer

    def fit(self, X, y, epochs=1000, lr=1e-3,
            batch_size=64, val_split=0.1,
            val_every=1, log_every=100, verbose=True,
            num_workers=0, checkpoint_dir=None, history_path=None,
            ckpt_path=None, gradient_clip_val=None,
            gradient_clip_algorithm='norm',
            early_stopping_patience=None, lr_scheduler=False,
            lr_scheduler_patience=10, lr_scheduler_factor=0.1):
        """
        Обучение модели (совместимость с предыдущим API)

        Args:
            gradient_clip_val (float, optional): Значение для клиппирования градиентов. По умолчанию None.
            gradient_clip_algorithm (str, optional): Алгоритм клиппирования ('norm' или 'value'). По умолчанию 'norm'.
            early_stopping_patience (int, optional): Количество эпох без улучшения для ранней остановки. По умолчанию None.
            lr_scheduler (bool, optional): Использовать scheduler обучения. По умолчанию False.
            lr_scheduler_patience (int): Терпение для ReduceLROnPlateau. По умолчанию 10.
            lr_scheduler_factor (float): Коэффициент уменьшения LR. По умолчанию 0.1.
        """
        self.lr = lr

        # Сохраняем параметры scheduler в гиперпараметры для configure_optimizers
        if lr_scheduler:
            self.hparams.lr_scheduler = True
            self.hparams.lr_scheduler_patience = lr_scheduler_patience
            self.hparams.lr_scheduler_factor = lr_scheduler_factor

        # Автоматический поиск последнего чекпоинта
        if ckpt_path is None and checkpoint_dir is not None:
            checkpoint_dir_path = Path(checkpoint_dir)
            if checkpoint_dir_path.exists():
                # Ищем чекпоинты с паттерном epoch=*.ckpt
                checkpoint_files = list(checkpoint_dir_path.glob("epoch=*.ckpt"))
                if checkpoint_files:
                    # Сортируем по времени модификации и берем последний
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

        # Добавляем LearningRateMonitor если используется scheduler
        if lr_scheduler:
            callbacks.append(LearningRateMonitor(logging_interval='epoch'))

        checkpoint_callback = None
        if val_split > 0:
            checkpoint_kwargs = {
                'monitor': 'val_loss',
                'mode': 'min',
                'save_top_k': 3,  # Сохраняем последние 3 лучших чекпоинта
                'verbose': verbose,
                'save_last': True  # Всегда сохраняем последнюю модель
            }
            if checkpoint_dir is not None:
                checkpoint_kwargs['dirpath'] = checkpoint_dir
            checkpoint_callback = ModelCheckpoint(**checkpoint_kwargs)
            callbacks.append(checkpoint_callback)

        # Добавляем EarlyStopping если задано терпение
        if early_stopping_patience is not None and val_split > 0:
            callbacks.append(EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                mode='min',
                verbose=verbose
            ))

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
            enable_checkpointing=checkpoint_dir is not None  # Отключаем чекпоинты если не указан путь
        )

        trainer.fit(self, train_dataloader, val_dataloader, ckpt_path=ckpt_path)

        if history_path is not None:
            history = history_callback.get_history()
            history_path = Path(history_path)
            history_path.parent.mkdir(parents=True, exist_ok=True)
            with open(history_path, 'w', encoding='utf-8') as f:
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
                state_dict = checkpoint['state_dict']
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    if k.startswith('model.'):
                        new_state_dict[k[6:]] = v  # удаляем 'model.'
                    else:
                        new_state_dict[k] = v
                self.load_state_dict(new_state_dict, strict=True)
        else:
            self.logger_py.info("Готово.")

    def predict(self, X):
        """Предсказание (возвращает numpy на CPU)"""
        self.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
            u = X_tensor[:, :self.n_var]
            p = X_tensor[:, self.n_var:self.n_var + self.n_param]
            return self.forward(u, p).cpu().numpy()

    def simulate(self, u0, p, n_steps, verbose=True, divergence_threshold=1e5):
        """Интегрирование ОДУ с опциональным прогресс‑баром и проверкой divergence"""
        u0 = np.atleast_2d(u0)
        p  = np.atleast_2d(p)

        trajectory = [u0.copy()]
        u_current = u0.copy()

        iterator = range(n_steps)
        if verbose:
            iterator = tqdm(iterator, desc='Симуляция', unit='шаг', ncols=100)

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
        """
        Универсальная загрузка модели из любого формата checkpoint

        Args:
            path: путь к файлу (.ckpt, .pt, .pth)
            device: 'cpu', 'cuda', или None (автовыбор)

        Returns:
            NeuroMap1: загруженная модель

        Raises:
            FileNotFoundError: файл не найден
            ValueError: файл поврежден или некорректен
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Файл не найден: {path.absolute()}")

        # Автовыбор device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        checkpoint = torch.load(str(path), map_location=device, weights_only=False)

        # 1. Lightning checkpoint (ModelCheckpoint)
        if 'pytorch-lightning_version' in checkpoint:
            return cls.load_from_checkpoint(str(path), map_location=device)

        # 2. Ручное сохранение через model.save()
        if '_format' in checkpoint and checkpoint['_format'] == 'neuromap_v1.0':
            hparams = checkpoint.get('hyper_parameters', {})
            required_params = ['n_var', 'n_param', 'hidden_size', 'dt', 'lr']
            missing = [p for p in required_params if p not in hparams]
            if missing:
                raise ValueError(
                    f"Отсутствуют обязательные гиперпараметры: {missing}"
                )
            hparams.setdefault('num_hidden_layers', 1)
            model = cls(**hparams)
            # Загружаем state_dict, игнорируя лишние ключи (например, буферы нормализации из старых версий)
            try:
                model.load_state_dict(checkpoint['state_dict'], strict=False)
            except RuntimeError as e:
                raise ValueError(
                    f"Не удалось загрузить state_dict. Ошибка: {str(e)}"
                ) from e
            model.to(device)

            # Попытка загрузить историю обучения из файла, если она не в checkpoint
            if not hasattr(model, 'training_history') or model.training_history is None:
                history_path = path.parent / f"{path.stem}_history.json"
                if history_path.exists():
                    try:
                        with open(history_path, 'r', encoding='utf-8') as f:
                            model.training_history = json.load(f)
                    except Exception as e:
                        model.logger_py.warning(f"Не удалось загрузить историю из {history_path}: {e}")

            return model

        # 3. Legacy или поврежденный файл
        raise ValueError(
            f"Файл {path} имеет неподдерживаемый формат. "
            f"Доступные ключи: {list(checkpoint.keys())[:10]}... "
            f"Используйте model.save() для сохранения в правильном формате."
        )

    def save(self, path, save_history=True):
        """
        Надежное сохранение модели в универсальном формате

        Args:
            path: путь к файлу
            save_history: сохранять ли историю в отдельный JSON

        Returns:
            Path: путь к сохраненному файлу
        """
        path = Path(path).with_suffix('.ckpt')
        path.parent.mkdir(parents=True, exist_ok=True)

        # Явные гиперпараметры
        hparams = {
            'n_var': int(self.n_var),
            'n_param': int(self.n_param),
            'hidden_size': int(self.hidden_size),
            'num_hidden_layers': int(self.num_hidden_layers),
            'dt': float(self.dt),
            'lr': float(self.lr)
        }

        # Сохраняем только state_dict (буферы нормализации больше не используются)
        state_dict = self.state_dict()

        checkpoint = {
            'state_dict': state_dict,
            'hyper_parameters': hparams,
            '_format': 'neuromap_v1.0',
            'torch_version': torch.__version__,
            'saved_at': datetime.now().isoformat()
        }

        torch.save(checkpoint, str(path))

        # Сохранение истории
        if save_history and hasattr(self, 'training_history'):
            history_path = path.parent / f"{path.stem}_history.json"
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(self.training_history, f, indent=2)

        return path

    def _forward_hidden_activations(self, z):
        """Возвращает список активаций скрытых слоёв (после tanh) для вычисления якобиана."""
        if self.num_hidden_layers == 1:
            h = self.activation(self.hidden(z))
            return [h]
        activations = []
        h = z
        for layer in self.hidden_layers:
            h = self.activation(layer(h))
            activations.append(h)
        return activations

    def compute_d_and_jacobian(self, u, p):
        """
        Вычисление приращения d и его якобиана за один проход.

        Для NeuroMapOriginal: d = dt * g

        Args:
            u: состояние, shape (n_var,) или (1, n_var)
            p: параметры, shape (n_param,) или (1, n_param)

        Returns:
            d: приращение, shape (n_var,)
            J_d: якобиан dd/du, shape (n_var, n_var)
        """
        self.eval()

        u_tensor = torch.tensor(u, dtype=torch.float32, device=self.device)
        p_tensor = torch.tensor(p, dtype=torch.float32, device=self.device)

        if u_tensor.dim() == 1:
            u_tensor = u_tensor.unsqueeze(0)
        if p_tensor.dim() == 1:
            p_tensor = p_tensor.unsqueeze(0)

        # Нормализация удалена – входные данные подаются как есть
        z = torch.cat([u_tensor, p_tensor], dim=1)
        activations = self._forward_hidden_activations(z)
        g = self.output(activations[-1])

        # d = dt * g
        d = self.dt * g

        # Якобиан: цепочка по слоям (без деления на стандартное отклонение)
        if self.num_hidden_layers == 1:
            # One hidden layer: J_d = dt * W_output @ diag(h') @ W_u
            h_derivative = 1 - activations[0]**2
            W_hidden = self.hidden.weight          # (hidden_size, n_var + n_param)
            W_u = W_hidden[:, :self.n_var]         # (hidden_size, n_var)
            W_output = self.output.weight          # (n_var, hidden_size)
            J_d = self.dt * (
                W_output @ torch.diag(h_derivative.squeeze()) @ W_u
            )
        else:
            # Multi‑layer: chain rule through all hidden layers
            W_hidden0 = self.hidden.weight          # (hidden_size, n_var + n_param)
            W_u0 = W_hidden0[:, :self.n_var]        # (hidden_size, n_var)
            h_deriv0 = 1 - activations[0]**2
            d_h_du = torch.diag(h_deriv0.squeeze()) @ W_u0
            # Propagate through remaining hidden layers
            for l in range(len(self.hidden_layers)):
                W_l = self.hidden_layers[l].weight    # (hidden_size, hidden_size)
                h_deriv_l = 1 - activations[l+1]**2
                d_h_du = torch.diag(h_deriv_l.squeeze()) @ W_l @ d_h_du
            # Output layer
            W_output = self.output.weight              # (n_var, hidden_size)
            J_d = self.dt * (W_output @ d_h_du)

        return d.squeeze().cpu().detach().numpy(), J_d.cpu().detach().numpy()

    def find_fixed_point(self, p, u0, tol=1e-10, method='hybr'):
        """
        Поиск неподвижной точки через scipy.optimize.root.

        Args:
            p: параметры системы, shape (n_param,)
            u0: начальное приближение, shape (n_var,)
            tol: точность сходимости
            method: метод оптимизации ('hybr', 'lm', 'broyden1', 'anderson')

        Returns:
            u_star: неподвижная точка, shape (n_var,)
            converged: True если сошлось
            result: полный результат scipy для диагностики
        """
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

    def find_all_fixed_points(self, p, bounds, n_grid=10,
                              tol=1e-10, unique_tol=1e-6, method='hybr'):
        """
        Поиск всех неподвижных точек в заданной области.

        Args:
            p: параметры системы, shape (n_param,)
            bounds: границы поиска, list of (min, max) для каждой переменной
            n_grid: число точек сетки по каждому измерению
            tol: точность сходимости
            unique_tol: порог для определения уникальности точек
            method: метод scipy.optimize.root ('hybr', 'lm', 'broyden1')

        Returns:
            fixed_points: список уникальных неподвижных точек
            multipliers: список мультипликаторов для каждой точки
        """
        grids = [np.linspace(b[0], b[1], n_grid) for b in bounds]
        mesh = np.meshgrid(*grids, indexing='ij')
        initial_guesses = np.stack([m.ravel() for m in mesh], axis=1)

        found_points = []

        for u0 in initial_guesses:
            u_star, converged, _ = self.find_fixed_point(p, u0,
                                                          tol=tol, method=method)

            if not converged:
                continue

            in_bounds = all(bounds[i][0] - unique_tol <= u_star[i] <= bounds[i][1] + unique_tol
                            for i in range(len(bounds)))
            if not in_bounds:
                continue

            is_unique = True
            for existing in found_points:
                if np.linalg.norm(u_star - existing) < unique_tol:
                    is_unique = False
                    break

            if is_unique:
                found_points.append(u_star)

        multipliers = [self.compute_fixed_point_multipliers(fp, p)
                       for fp in found_points]

        return found_points, multipliers

    def compute_fixed_point_multipliers(self, u_star, p):
        """
        Вычисление мультипликаторов неподвижной точки.

        Args:
            u_star: неподвижная точка (можно найти через find_fixed_point)
            p: параметры системы

        Returns:
            eigenvalues: собственные значения якобиана отображения (мультипликаторы)
        """
        self.eval()

        u_star_tensor = torch.tensor(u_star, dtype=torch.float32,
                                     device=self.device)
        p_tensor = torch.tensor(p, dtype=torch.float32,
                                device=self.device)

        if u_star_tensor.dim() == 1:
            u_star_tensor = u_star_tensor.unsqueeze(0)
        if p_tensor.dim() == 1:
            p_tensor = p_tensor.unsqueeze(0)

        batch_size = u_star_tensor.shape[0]

        # Нормализация удалена – входные данные подаются как есть
        z = torch.cat([u_star_tensor, p_tensor], dim=1)
        activations = self._forward_hidden_activations(z)

        I = torch.eye(self.n_var, device=self.device)
        multipliers = []

        for i in range(batch_size):
            if self.num_hidden_layers == 1:
                h_derivative = 1 - activations[0][i]**2
                W_hidden = self.hidden.weight
                W_u = W_hidden[:, :self.n_var]
                W_output = self.output.weight
                J_d = self.dt * (
                    W_output @ torch.diag(h_derivative) @ W_u
                )
            else:
                W_hidden0 = self.hidden.weight
                W_u0 = W_hidden0[:, :self.n_var]
                h_deriv0 = 1 - activations[0][i]**2
                d_h_du = torch.diag(h_deriv0) @ W_u0
                for l in range(len(self.hidden_layers)):
                    W_l = self.hidden_layers[l].weight
                    h_deriv_l = 1 - activations[l+1][i]**2
                    d_h_du = torch.diag(h_deriv_l) @ W_l @ d_h_du
                dg_du = self.output.weight @ d_h_du
                J_d = self.dt * (dg_du)

            J_n = I + J_d
            eigenvalues = torch.linalg.eigvals(J_n)
            multipliers.append(eigenvalues.cpu().detach().numpy())

        if batch_size == 1:
            return multipliers[0]
        return multipliers
