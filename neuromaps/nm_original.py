import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
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
            'epoch': []
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
        elif self.last_val_loss is not None:
            self.history['val_loss'].append(self.last_val_loss)
        
        self.history['epoch'].append(epoch)
    
    def get_history(self):
        """Возвращает историю обучения"""
        return self.history.copy()


class NeuroMapOriginal(pl.LightningModule):
    """Модель С ОШИБКОЙ из статьи: денормализация по u вместо d"""
    
    def __init__(self, n_var, n_param, hidden_size=50, dt=0.01, lr=1e-3):
        super().__init__()
        self.save_hyperparameters('n_var', 'n_param', 'hidden_size', 'dt', 'lr')
        
        self.n_var = n_var
        self.n_param = n_param
        self.hidden_size = hidden_size
        self.dt = dt
        self.lr = lr
        
        self.logger_py = get_logger(__name__)
        
        # === Только статистика для входов, НЕТ статистики для приращений ===
        self.register_buffer('mu_u', torch.zeros(1, n_var))
        self.register_buffer('su', torch.ones(1, n_var))
        self.register_buffer('mu_p', torch.zeros(1, n_param))
        self.register_buffer('sp', torch.ones(1, n_param))
        # mu_d и sd УДАЛЕНЫ - их нет в статье
        
        self.hidden = nn.Linear(n_var + n_param, hidden_size)
        self.activation = nn.Tanh()
        self.output = nn.Linear(hidden_size, n_var)
        
        self.criterion = nn.MSELoss()
    
    def forward(self, u, p):
        """Прямой проход С ОШИБКОЙ"""
        u_norm = (u - self.mu_u) / self.su
        p_norm = (p - self.mu_p) / self.sp
        
        z = torch.cat([u_norm, p_norm], dim=1)
        h = self.activation(self.hidden(z))
        g = self.output(h)
        
        # === Используем статистику u для денормализации d ===
        d = self.dt * (g * self.su + self.mu_u)
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
            val_every=10, log_every=100, verbose=True, num_workers=0, 
            checkpoint_dir=None, history_path=None):
        """
        Обучение модели (совместимость с предыдущим API)
        """
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
        
        history_callback = HistoryCallback()
        callbacks = [history_callback, TQDMProgressBar(refresh_rate=1 if verbose else 0)]
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
        
        log_every_n_steps = log_every if log_every > 0 else 50
        
        trainer = pl.Trainer(
            max_epochs=epochs,
            callbacks=callbacks,
            enable_progress_bar=verbose,
            log_every_n_steps=log_every_n_steps,
            check_val_every_n_epoch=val_every if val_split > 0 else 1,
            logger=False
        )
        
        trainer.fit(self, train_dataloader, val_dataloader)
        
        if history_path is not None:
            history = history_callback.get_history()
            history_path = Path(history_path)
            history_path.parent.mkdir(parents=True, exist_ok=True)
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2, ensure_ascii=False)
            self.logger_py.info(f"История обучения сохранена в {history_path}")
        
        self.training_history = history_callback.get_history()
        
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
        """Вычисление статистики С ОШИБКОЙ (не учитывает масштаб y)"""
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
        
        # === Статистика для d НЕ вычисляется ===
        update_stat(self.mu_u, u)
        update_stat(self.su, u, is_std=True)
        update_stat(self.mu_p, p)
        update_stat(self.sp, p, is_std=True)
        # mu_d и sd ОТСУТСТВУЮТ
    
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
            
            model = cls(**hparams)
            try:
                model.load_state_dict(checkpoint['state_dict'], strict=True)
            except RuntimeError as e:
                raise ValueError(
                    f"Не удалось загрузить state_dict. Возможно, модель была сохранена с другой архитектурой. "
                    f"Ошибка: {str(e)}"
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
            'dt': float(self.dt),
            'lr': float(self.lr)
        }
        
        checkpoint = {
            'state_dict': self.state_dict(),
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
