import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from utils.logger import get_logger

class NeuroMap1(nn.Module):
    
    def __init__(self, n_var, n_param, hidden_size=50, dt=0.01, device=None):
        super().__init__()
        self.n_var = n_var
        self.n_param = n_param
        self.hidden_size = hidden_size
        self.dt = dt
        
        self.logger = get_logger(__name__)
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.logger.info(f"device: {self.device}")
        
        self.register_buffer('mu_u', torch.zeros(1, n_var))
        self.register_buffer('su', torch.ones(1, n_var))
        self.register_buffer('mu_p', torch.zeros(1, n_param))
        self.register_buffer('sp', torch.ones(1, n_param))
        self.register_buffer('mu_d', torch.zeros(1, n_var))
        self.register_buffer('sd', torch.ones(1, n_var))
        
        self.hidden = nn.Linear(n_var + n_param, hidden_size)
        self.activation = nn.Tanh()
        self.output = nn.Linear(hidden_size, n_var)
        
        self.to(self.device)
    
    def forward(self, u, p):
        u_norm = (u - self.mu_u) / self.su
        p_norm = (p - self.mu_p) / self.sp
        
        z = torch.cat([u_norm, p_norm], dim=1)
        h = self.activation(self.hidden(z))
        g = self.output(h)
        
        d = self.dt * (g * self.sd + self.mu_d)
        return d
    
    def fit(self, X, y, epochs=1000, lr=1e-3, batch_size=64, val_split=0.1,
        val_every=10, log_every=100, verbose=True):
        self._compute_statistics(X, y)

        N = len(X)
        n_train = int(N * (1 - val_split))
        X_train, X_val = X[:n_train], X[n_train:]
        y_train, y_val = y[:n_train], y[n_train:]

        self.logger.info(f"Тренировка: {len(X_train):,} | Валидация: {len(X_val):,}")

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32, device=self.device)

        dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            pin_memory=(self.device.type == 'cuda')
        )

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()

        best_val_loss = float('inf')
        epoch_bar = tqdm(range(epochs), desc='Обучение', unit='эпоха',
                        ncols=120, disable=not verbose, position=0)

        for epoch in epoch_bar:
            self.train()
            train_loss = 0.0

            for X_batch, y_batch in dataloader:
                u_batch = X_batch[:, :self.n_var]
                p_batch = X_batch[:, self.n_var:self.n_var + self.n_param]

                optimizer.zero_grad()
                pred = self.forward(u_batch, p_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Валидация
            val_loss = None
            if val_split > 0 and epoch % val_every == 0:
                self.eval()
                with torch.no_grad():
                    X_val_tensor = torch.tensor(X_val, dtype=torch.float32, device=self.device)
                    y_val_tensor = torch.tensor(y_val, dtype=torch.float32, device=self.device)
                    u_val = X_val_tensor[:, :self.n_var]
                    p_val = X_val_tensor[:, self.n_var:self.n_var + self.n_param]

                    val_pred = self.forward(u_val, p_val)
                    val_loss = criterion(val_pred, y_val_tensor).item()

                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        self.logger.debug(f'Лучшая валидация: {best_val_loss:.6f}')

            avg_train_loss = train_loss / len(dataloader)
            postfix = {'loss': f'{avg_train_loss:.6f}'}
            if val_loss is not None:
                postfix['val_loss'] = f'{val_loss:.6f}'
            epoch_bar.set_postfix(postfix)

            if epoch % log_every == 0:
                self.logger.debug(f"Эпоха {epoch:4d} | {epoch_bar.format_dict['postfix']}")

        epoch_bar.close()
        self.logger.info(f"Готово. Лучшая валидация: {best_val_loss:.6f}")
        
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
            self.logger.warning("Обнаружены константные приращения (std < 1e-8)")
        
        def update_stat(buffer, data, is_std=False):
            tensor = torch.tensor(np.mean(data, axis=0, keepdims=True) if not is_std else np.std(data, axis=0, keepdims=True), 
                                 dtype=torch.float32, device=self.device)
            buffer.copy_(tensor)
            if is_std:
                buffer.clamp_(min=1e-6)
        
        update_stat(self.mu_u, u)
        update_stat(self.su, u, is_std=True)
        update_stat(self.mu_p, p)
        update_stat(self.sp, p, is_std=True)
        update_stat(self.mu_d, d)
        update_stat(self.sd, d, is_std=True)
