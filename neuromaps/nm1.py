import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar

from utils.logger import get_logger


class NeuroMap1(pl.LightningModule):
    
    def __init__(self, n_var, n_param, hidden_size=50, dt=0.01, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        self.n_var = n_var
        self.n_param = n_param
        self.hidden_size = hidden_size
        self.dt = dt
        self.lr = lr
        
        self.logger_py = get_logger(__name__)
        
        self.register_buffer('mu_u', torch.zeros(1, n_var))
        self.register_buffer('su', torch.ones(1, n_var))
        self.register_buffer('mu_p', torch.zeros(1, n_param))
        self.register_buffer('sp', torch.ones(1, n_param))
        self.register_buffer('mu_d', torch.zeros(1, n_var))
        self.register_buffer('sd', torch.ones(1, n_var))
        
        self.hidden = nn.Linear(n_var + n_param, hidden_size)
        self.activation = nn.Tanh()
        self.output = nn.Linear(hidden_size, n_var)
        
        self.criterion = nn.MSELoss()
    
    def forward(self, u, p):
        u_norm = (u - self.mu_u) / self.su
        p_norm = (p - self.mu_p) / self.sp
        
        z = torch.cat([u_norm, p_norm], dim=1)
        h = self.activation(self.hidden(z))
        g = self.output(h)
        
        d = self.dt * (g * self.sd + self.mu_d)
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
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def fit(self, X, y, epochs=1000, lr=1e-3, batch_size=64, val_split=0.1,
            val_every=10, log_every=100, verbose=True, num_workers=79, checkpoint_dir=None):
        """Обучение модели (совместимость с предыдущим API)"""
        self.lr = lr
        self._compute_statistics(X, y)

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
        
        # Настройка callbacks
        callbacks = [TQDMProgressBar(refresh_rate=1 if verbose else 0)]
        checkpoint_callback = None
        if val_split > 0:
            checkpoint_kwargs = {
                'monitor': 'val_loss',
                'mode': 'min',
                'save_top_k': 1,
                'verbose': verbose
            }
            if checkpoint_dir is not None:
                checkpoint_kwargs['dirpath'] = checkpoint_dir
            checkpoint_callback = ModelCheckpoint(**checkpoint_kwargs)
            callbacks.append(checkpoint_callback)
        
        # Настройка логирования
        log_every_n_steps = log_every if log_every > 0 else 50
        
        trainer = pl.Trainer(
            max_epochs=epochs,
            callbacks=callbacks,
            enable_progress_bar=verbose,
            log_every_n_steps=log_every_n_steps,
            check_val_every_n_epoch=val_every if val_split > 0 else 1,
            logger=False  # Используем наш собственный логгер
        )
        
        trainer.fit(self, train_dataloader, val_dataloader)
        
        if checkpoint_callback is not None and checkpoint_callback.best_model_score is not None:
            self.logger_py.info(f"Готово. Лучшая валидация: {checkpoint_callback.best_model_score:.6f}")
        else:
            self.logger_py.info("Готово.")
        
    def predict(self, X):
        """Предсказание (возвращает numpy на CPU)"""
        self.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
            u = X_tensor[:, :self.n_var]
            p = X_tensor[:, self.n_var:self.n_var+self.n_param]
            return self.forward(u, p).cpu().numpy()
    
    def simulate(self, u0, p, n_steps):
        """Интегрирование ОДУ с прогресс-баром"""
        u0 = np.atleast_2d(u0)
        p  = np.atleast_2d(p)
        
        trajectory = [u0.copy()]
        u_current = u0.copy()

        for _ in tqdm(range(n_steps), desc='Симуляция', unit='шаг', ncols=100):
            X_step = np.concatenate([u_current, p], axis=1)
            d = self.predict(X_step)
            u_current = u_current + d
            trajectory.append(u_current.copy())

        return np.concatenate(trajectory, axis=0)
    
    def _compute_statistics(self, X, y):
        """Вычисление статистики с валидацией"""
        u = X[:, :self.n_var]
        p = X[:, self.n_var:self.n_var+self.n_param]
        d = y
        
        if np.any(np.std(d, axis=0) < 1e-8):
            self.logger_py.warning("Обнаружены константные приращения (std < 1e-8)")
        
        def update_stat(buffer, data, is_std=False):
            tensor = torch.tensor(
                np.mean(data, axis=0, keepdims=True) if not is_std else np.std(data, axis=0, keepdims=True), 
                dtype=torch.float32
            )
            buffer.copy_(tensor)
            if is_std:
                buffer.clamp_(min=1e-6)
        
        update_stat(self.mu_u, u)
        update_stat(self.su, u, is_std=True)
        update_stat(self.mu_p, p)
        update_stat(self.sp, p, is_std=True)
        update_stat(self.mu_d, d)
        update_stat(self.sd, d, is_std=True)
    
    def save(self, path):
        """
        Сохранение модели в файл (совместимость с предыдущим API)
        
        Args:
            path: путь к файлу для сохранения (str или Path)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Сохраняем checkpoint в формате Lightning
        # Преобразуем hparams в словарь, если это Namespace
        hparams = self.hparams
        if hasattr(hparams, '__dict__'):
            hparams = hparams.__dict__
        elif not isinstance(hparams, dict):
            hparams = dict(hparams)
        
        checkpoint = {
            'state_dict': self.state_dict(),
            'hyper_parameters': hparams
        }
        torch.save(checkpoint, str(path))
        self.logger_py.info(f"Модель сохранена в {path}")
    
    @classmethod
    def load(cls, path, device=None):
        """
        Загрузка модели из файла
        
        Args:
            path: путь к файлу с сохраненной моделью (str или Path)
            device: устройство для загрузки модели (если None, определяется автоматически)
        
        Returns:
            NeuroMap1: загруженная модель
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Файл модели не найден: {path}")
        
        # Загружаем чекпоинт
        checkpoint = torch.load(str(path), map_location=device)
        
        # Проверяем формат чекпоинта
        if 'pytorch-lightning_version' in checkpoint:
            # Полный формат Lightning - используем встроенный метод
            model = cls.load_from_checkpoint(str(path), map_location=device)
        else:
            # Кастомный формат - загружаем вручную
            hparams = checkpoint.get('hyper_parameters', {})
            if not isinstance(hparams, dict):
                hparams = dict(hparams)
            
            # Создаем модель с гиперпараметрами
            model = cls(**hparams)
            
            # Загружаем веса
            model.load_state_dict(checkpoint['state_dict'])
            
            if device is not None:
                model.to(device)
        
        logger = get_logger(__name__)
        logger.info(f"Модель загружена из {path}")
        
        return model
