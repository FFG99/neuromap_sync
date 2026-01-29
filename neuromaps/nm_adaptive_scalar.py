import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor
)
from pytorch_lightning.callbacks.progress import TQDMProgressBar
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


class NeuroMapAdaptiveScale(pl.LightningModule):
    """
    NeuroMap с адаптивным масштабированием выхода.
    
    Ключевая особенность: масштаб и сдвиг выхода — обучаемые параметры,
    а не статистики таргета. Это позволяет модели автоматически подстроиться
    под любую динамику.
    
    Формула предсказания:
        Δu = scale * g(u_norm, p_norm) + bias
    
    где:
        - g — нелинейная функция, выученная сетью (выход в [-1, 1])
        - scale, bias — обучаемые параметры (векторные, по одному на компоненту)
        - Нормализация применяется ТОЛЬКО к входу (через статистики u и p)
    """
    
    def __init__(self, n_var, n_param, hidden_size=50, lr=1e-3, scale_init=0.01):
        """
        Args:
            n_var: число переменных состояния
            n_param: число параметров системы
            hidden_size: размер скрытого слоя
            lr: learning rate
            scale_init: начальное значение масштаба (рекомендуется ~Δt для ОДУ)
        """
        super().__init__()
        self.save_hyperparameters('n_var', 'n_param', 'hidden_size', 'lr', 'scale_init')
        
        self.n_var = n_var
        self.n_param = n_param
        self.hidden_size = hidden_size
        self.lr = lr
        
        self.logger_py = get_logger(__name__)
        
        # Статистики ВХОДА (только для нормализации входа!)
        self.register_buffer('mu_u', torch.zeros(1, n_var))
        self.register_buffer('su', torch.ones(1, n_var))
        self.register_buffer('mu_p', torch.zeros(1, n_param))
        self.register_buffer('sp', torch.ones(1, n_param))
        
        # Скрытые слои
        self.hidden = nn.Linear(n_var + n_param, hidden_size)
        self.activation = nn.Tanh()
        self.output = nn.Linear(hidden_size, n_var)
        
        # ОБУЧАЕМЫЕ ПАРАМЕТРЫ МАСШТАБИРОВАНИЯ (ключевое отличие от оригинала!)
        # Инициализируем как ~Δt для непрерывных систем, но модель сама подстроит
        self.scale = nn.Parameter(torch.ones(n_var) * scale_init)
        self.bias = nn.Parameter(torch.zeros(n_var))  # для асимметричных распределений Δu
        
        self.criterion = nn.MSELoss()
    
    def forward(self, u, p):
        """
        Предсказывает приращение Δu для заданного состояния и параметров.
        
        Важно: нормализация применяется ТОЛЬКО к входу.
        Масштабирование выхода происходит через обучаемые параметры scale/bias.
        """
        u_norm = (u - self.mu_u) / self.su
        p_norm = (p - self.mu_p) / self.sp
        
        z = torch.cat([u_norm, p_norm], dim=1)
        h = self.activation(self.hidden(z))
        g = self.output(h)  # g ∈ [-1, 1] при tanh в скрытых слоях
        
        d = g * self.scale + self.bias
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
    
    def fit(self, X, y, epochs=1000, lr=1e-3, batch_size=64, val_split=0.1,
            val_every=10, log_every=100, verbose=True, num_workers=0, checkpoint_dir=None,
            history_path=None, ckpt_path=None, gradient_clip_val=None, gradient_clip_algorithm='norm',
            early_stopping_patience=None, lr_scheduler=False, lr_scheduler_patience=10, lr_scheduler_factor=0.1):
        """
        Обучение модели с адаптивным масштабированием.
        
        Важно: статистики таргета (Δu) НЕ используются в архитектуре —
        только для информации (логгирования). Масштаб подстраивается обучаемыми параметрами.
        """
        self.lr = lr
        self._compute_statistics(X, y)
        
        if lr_scheduler:
            self.hparams.lr_scheduler = True
            self.hparams.lr_scheduler_patience = lr_scheduler_patience
            self.hparams.lr_scheduler_factor = lr_scheduler_factor
        
        # Автоматический поиск последнего чекпоинта
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
            callbacks.append(LearningRateMonitor(logging_interval='epoch'))
        
        checkpoint_callback = None
        if val_split > 0:
            checkpoint_kwargs = {
                'monitor': 'val_loss',
                'mode': 'min',
                'save_top_k': 3,
                'verbose': verbose,
                'save_last': True
            }
            if checkpoint_dir is not None:
                checkpoint_kwargs['dirpath'] = checkpoint_dir
            checkpoint_callback = ModelCheckpoint(**checkpoint_kwargs)
            callbacks.append(checkpoint_callback)
        
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
            enable_checkpointing=checkpoint_dir is not None
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
        else:
            self.logger_py.info("Готово.")
        
        if checkpoint_callback is not None and checkpoint_callback.best_model_path:
            self.logger_py.info(f"Загружаем лучшую модель из {checkpoint_callback.best_model_path}")
            checkpoint = torch.load(checkpoint_callback.best_model_path, map_location=self.device)
            state_dict = checkpoint['state_dict']
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith('model.'):
                    new_state_dict[k[6:]] = v
                else:
                    new_state_dict[k] = v
            self.load_state_dict(new_state_dict, strict=True)
    
    def predict(self, X):
        """Предсказание (возвращает numpy на CPU)"""
        self.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
            u = X_tensor[:, :self.n_var]
            p = X_tensor[:, self.n_var:self.n_var+self.n_param]
            return self.forward(u, p).cpu().numpy()
    
    def simulate(self, u0, p, n_steps, verbose=True, divergence_threshold=1e5):
        """Интегрирование ОДУ с прогресс-баром и проверкой на расходимость"""
        u0 = np.atleast_2d(u0)
        p  = np.atleast_2d(p)
        
        trajectory = [u0.copy()]
        u_current = u0.copy()

        iterator = range(n_steps)
        if verbose:
            iterator = tqdm(iterator, desc='Симуляция', unit='шаг', ncols=100, disable=False)
        else:
            iterator = tqdm(iterator, desc='Симуляция', unit='шаг', ncols=100, disable=True)

        for _ in iterator:
            X_step = np.concatenate([u_current, p], axis=1)
            d = self.predict(X_step)
            u_current = u_current + d
            
            if np.linalg.norm(u_current) > divergence_threshold:
                self.logger_py.info(f"Траектория разошлась (норма = {np.linalg.norm(u_current):.2e} > {divergence_threshold:.2e})")
                return None
            
            trajectory.append(u_current.copy())

        return np.concatenate(trajectory, axis=0)
    
    def _compute_statistics(self, X, y):
        """
        Вычисление статистик ТОЛЬКО для входа (u, p).
        Статистики таргета (Δu) вычисляются для информации, но НЕ используются в архитектуре.
        """
        u = X[:, :self.n_var]
        p = X[:, self.n_var:self.n_var+self.n_param]
        d = y  # приращения — для информации/логгирования
        
        # Логгируем статистики таргета для диагностики
        if np.any(np.std(d, axis=0) < 1e-8):
            self.logger_py.warning("Обнаружены константные приращения (std < 1e-8)")
        
        # Вычисляем отношение масштабов для диагностики
        su_empirical = np.std(u, axis=0)
        sd_empirical = np.std(d, axis=0) + 1e-8
        ratio = su_empirical / sd_empirical
        self.logger_py.info(f"Эмпирическое отношение масштабов s_u/s_Δu: {ratio}")
        self.logger_py.info(f"Инициализация scale: {self.scale.detach().cpu().numpy()}")
        self.logger_py.info("Модель сама подстроит масштаб через обучаемый параметр 'scale'")
        
        def update_stat(buffer, data, is_std=False):
            tensor = torch.tensor(
                np.mean(data, axis=0, keepdims=True) if not is_std else np.std(data, axis=0, keepdims=True), 
                dtype=torch.float32
            )
            buffer.copy_(tensor)
            if is_std:
                buffer.clamp_(min=1e-6)
        
        # Обновляем ТОЛЬКО статистики входа
        update_stat(self.mu_u, u)
        update_stat(self.su, u, is_std=True)
        update_stat(self.mu_p, p)
        update_stat(self.sp, p, is_std=True)
        
        # Статистики таргета НЕ регистрируем в буферах — они не используются в forward!
    
    @classmethod
    def load(cls, path, device=None):
        """Универсальная загрузка модели"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Файл не найден: {path.absolute()}")
        
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        checkpoint = torch.load(str(path), map_location=device, weights_only=False)
        
        # Lightning checkpoint
        if 'pytorch-lightning_version' in checkpoint:
            return cls.load_from_checkpoint(str(path), map_location=device)
        
        # Ручное сохранение
        if '_format' in checkpoint and checkpoint['_format'] == 'neuromap_v1.0':
            hparams = checkpoint.get('hyper_parameters', {})
            required_params = ['n_var', 'n_param', 'hidden_size', 'lr', 'scale_init']
            missing = [p for p in required_params if p not in hparams]
            if missing:
                raise ValueError(f"Отсутствуют обязательные гиперпараметры: {missing}")
            
            model = cls(**hparams)
            try:
                model.load_state_dict(checkpoint['state_dict'], strict=True)
            except RuntimeError as e:
                raise ValueError(
                    f"Не удалось загрузить state_dict. Возможно, модель была сохранена с другой архитектурой. "
                    f"Ошибка: {str(e)}"
                ) from e
            model.to(device)
            
            if not hasattr(model, 'training_history') or model.training_history is None:
                history_path = path.parent / f"{path.stem}_history.json"
                if history_path.exists():
                    try:
                        with open(history_path, 'r', encoding='utf-8') as f:
                            model.training_history = json.load(f)
                    except Exception as e:
                        model.logger_py.warning(f"Не удалось загрузить историю из {history_path}: {e}")
            
            return model
        
        raise ValueError(
            f"Файл {path} имеет неподдерживаемый формат. "
            f"Доступные ключи: {list(checkpoint.keys())[:10]}... "
            f"Используйте model.save() для сохранения в правильном формате."
        )

    def save(self, path, save_history=True):
        """Надежное сохранение модели в универсальном формате"""
        path = Path(path).with_suffix('.ckpt')
        path.parent.mkdir(parents=True, exist_ok=True)
        
        hparams = {
            'n_var': int(self.n_var),
            'n_param': int(self.n_param),
            'hidden_size': int(self.hidden_size),
            'lr': float(self.lr),
            'scale_init': float(self.scale.mean().item())  # сохраняем среднее для информации
        }
        
        state_dict = self.state_dict()
        
        checkpoint = {
            'state_dict': state_dict,
            'hyper_parameters': hparams,
            '_format': 'neuromap_v1.0',
            'torch_version': torch.__version__,
            'saved_at': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, str(path))
        
        if save_history and hasattr(self, 'training_history'):
            history_path = path.parent / f"{path.stem}_history.json"
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(self.training_history, f, indent=2)
        
        return path

    def compute_d_and_jacobian(self, u, p):
        """
        Вычисление приращения d и его якобиана.
        
        Важно: якобиан учитывает обучаемый масштаб:
            J_d = diag(scale) @ J_g
        где J_g — якобиан нелинейной функции g(u_norm, p_norm)
        """
        self.eval()
        
        u_tensor = torch.tensor(u, dtype=torch.float32, device=self.device)
        p_tensor = torch.tensor(p, dtype=torch.float32, device=self.device)
        
        if u_tensor.dim() == 1:
            u_tensor = u_tensor.unsqueeze(0)
        if p_tensor.dim() == 1:
            p_tensor = p_tensor.unsqueeze(0)
        
        u_norm = (u_tensor - self.mu_u) / self.su
        p_norm = (p_tensor - self.mu_p) / self.sp
        
        z = torch.cat([u_norm, p_norm], dim=1)
        
        h_linear = self.hidden(z)
        h_n = self.activation(h_linear)
        g = self.output(h_n)
        
        # Приращение с адаптивным масштабированием
        d = g * self.scale + self.bias
        
        # Якобиан: J_d = diag(scale) @ J_g
        # где J_g = (W_u.T / su.T) @ diag(h'(h_linear)) @ W_output.T
        h_derivative = 1 - h_n**2
        
        W_hidden = self.hidden.weight
        W_u = W_hidden[:, :self.n_var]
        W_output = self.output.weight
        
        # Якобиан нормализованного входа → скрытых активаций
        A0 = W_u.T / self.su.T  # [n_var, hidden_size]
        
        # Якобиан скрытых активаций → нормализованного выхода g
        A1 = W_output.T  # [hidden_size, n_var]
        
        H_n = torch.diag(h_derivative.squeeze())  # [hidden_size, hidden_size]
        
        # Якобиан нормализованного выхода g по нормализованному входу u_norm
        J_g = A0 @ H_n @ A1  # [n_var, n_var]
        
        # Якобиан финального выхода d по u (учитываем обучаемый масштаб)
        # d = scale * g + bias  =>  J_d = diag(scale) @ J_g
        J_d = torch.diag(self.scale) @ J_g
        
        return d.squeeze().cpu().detach().numpy(), J_d.cpu().detach().numpy()

    def find_fixed_point(self, p, u0, tol=1e-10, method='hybr'):
        """
        Поиск неподвижной точки: решение уравнения Δu = 0.
        
        Использует адаптивное масштабирование — не зависит от статистик таргета.
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

    def find_all_fixed_points(self, p, bounds, n_grid=10, tol=1e-10, unique_tol=1e-6, method='hybr'):
        """
        Поиск всех неподвижных точек в заданной области.
        """
        grids = [np.linspace(b[0], b[1], n_grid) for b in bounds]
        mesh = np.meshgrid(*grids, indexing='ij')
        initial_guesses = np.stack([m.ravel() for m in mesh], axis=1)
        
        found_points = []
        
        for u0 in initial_guesses:
            u_star, converged, _ = self.find_fixed_point(p, u0, tol=tol, method=method)
            
            if not converged:
                continue
            
            in_bounds = all(bounds[i][0] <= u_star[i] <= bounds[i][1] for i in range(len(bounds)))
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
        """
        Вычисление мультипликаторов неподвижной точки.
        
        Для отображения u_{t+1} = u_t + Δu:
            J_map = I + J_Δu
        где J_Δu — якобиан приращения (вычислен с учётом обучаемого масштаба).
        """
        self.eval()
        
        u_star_tensor = torch.tensor(u_star, dtype=torch.float32, device=self.device)
        p_tensor = torch.tensor(p, dtype=torch.float32, device=self.device)
        
        if u_star_tensor.dim() == 1:
            u_star_tensor = u_star_tensor.unsqueeze(0)
        if p_tensor.dim() == 1:
            p_tensor = p_tensor.unsqueeze(0)
        
        batch_size = u_star_tensor.shape[0]
        
        u_norm = (u_star_tensor - self.mu_u) / self.su
        p_norm = (p_tensor - self.mu_p) / self.sp
        
        z = torch.cat([u_norm, p_norm], dim=1)
        
        h_linear = self.hidden(z)
        h_n = self.activation(h_linear)
        
        h_derivative = 1 - h_n**2
        
        W_hidden = self.hidden.weight
        W_u = W_hidden[:, :self.n_var]
        W_output = self.output.weight
        
        A0 = W_u.T / self.su.T
        A1 = W_output.T  # без умножения на sd*dt!
        I = torch.eye(self.n_var, device=self.device)
        
        multipliers = []
        
        for i in range(batch_size):
            H_n_i = torch.diag(h_derivative[i])
            
            # Якобиан Δu с учётом обучаемого масштаба
            J_d = torch.diag(self.scale) @ (A0 @ H_n_i @ A1)
            
            # Якобиан отображения: u_{t+1} = u_t + Δu
            J_map = I + J_d
            
            eigenvalues = torch.linalg.eigvals(J_map)
            multipliers.append(eigenvalues.cpu().detach().numpy())
        
        if batch_size == 1:
            return multipliers[0]
        return multipliers
