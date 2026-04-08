import numpy as np
from tqdm import tqdm

class CoupledOscillators:
    def __init__(self, model1, model2):
        """
        Инициализация связанных осцилляторов.
        
        Args:
            model1: первый нейромэп (NeuroMapTargetNormalized или NeuroMapOriginal)
            model2: второй нейромэп (NeuroMapTargetNormalized или NeuroMapOriginal)
        """
        if not self._is_neuromap(model1):
            raise ValueError("model1 должен быть нейромэпом (NeuroMapTargetNormalized или NeuroMapOriginal)")
        if not self._is_neuromap(model2):
            raise ValueError("model2 должен быть нейромэпом (NeuroMapTargetNormalized или NeuroMapOriginal)")
        
        self.model1 = model1
        self.model2 = model2
        
        if model1.dt != model2.dt:
            raise ValueError(
                f"Модели имеют разные dt: {model1.dt} и {model2.dt}. "
                "Обе модели должны иметь одинаковый dt."
            )
    
    def _is_neuromap(self, obj):
        """Проверка, является ли объект нейроотображением"""
        return hasattr(obj, 'predict') and hasattr(obj, 'dt')
    
    def simulate(self, u0, p, n_steps, 
                 epsilon=0.1,
                 verbose=True):
        """
        Симуляция связанных осцилляторов как 4-мерной системы.
        
        Args:
            u0: начальное состояние системы [x1, ẋ1, x2, ẋ2] (4D)
            p: параметры системы [λ1, ω₀²₁, λ2, ω₀²₂] (4D)
            n_steps: количество шагов симуляции
            epsilon: параметр связи μ (сила связи между осцилляторами)
            verbose: показывать ли прогресс-бар
        
        Returns:
            np.ndarray: траектория системы формы (n_steps+1, 4)
                        каждая строка: [x1, ẋ1, x2, ẋ2]
        """
        u0 = np.atleast_2d(u0)
        p = np.atleast_2d(p)
        
        if u0.shape[1] != 4:
            raise ValueError(f"u0 должен иметь 4 компоненты, получено {u0.shape[1]}")
        if p.shape[1] != 4:
            raise ValueError(f"p должен иметь 4 компоненты, получено {p.shape[1]}")
        
        u1_current = u0[:, [0, 1]].copy()  # [x1, ẋ1]
        u2_current = u0[:, [2, 3]].copy()  # [x2, ẋ2]
        p1 = p[:, [0, 1]].copy()  # [λ1, ω₀²₁]
        p2 = p[:, [2, 3]].copy()  # [λ2, ω₀²₂]
        
        trajectory = [u0.copy()]
        
        iterator = range(n_steps)
        if verbose:
            iterator = tqdm(iterator, desc='Симуляция связанных осцилляторов', 
                          unit='шаг', ncols=100, disable=False)
        else:
            iterator = tqdm(iterator, desc='Симуляция связанных осцилляторов', 
                          unit='шаг', ncols=100, disable=True)
        
        for _ in iterator:
            # Вычисляем приращения от каждой модели
            X1_step = np.concatenate([u1_current, p1], axis=1)
            d1 = self.model1.predict(X1_step)
            
            X2_step = np.concatenate([u2_current, p2], axis=1)
            d2 = self.model2.predict(X2_step)
            
            # Добавляем член связи
            coupling1, coupling2 = self._compute_coupling(
                u1_current, u2_current, epsilon
            )
            
            dt = self.model1.dt  # dt одинаковый для обеих моделей
            d1 = d1 + coupling1 * dt
            d2 = d2 + coupling2 * dt
            
            # Обновляем состояния
            u1_current = u1_current + d1
            u2_current = u2_current + d2
            
            # Конкатенируем в 4D состояние
            u_current = np.concatenate([u1_current, u2_current], axis=1)
            trajectory.append(u_current.copy())
        
        return np.concatenate(trajectory, axis=0)
    
    def _compute_coupling(self, u1, u2, epsilon):
        """
        Вычисление членов связи между осцилляторами.
        
        Args:
            u1: состояние первого осциллятора [x1, ẋ1]
            u2: состояние второго осциллятора [x2, ẋ2]
            epsilon: параметр связи μ
        
        Returns:
            tuple: (coupling1, coupling2) - векторы связи для каждого осциллятора
                   coupling[i] добавляется к приращению d[i]
        """
        x1, v1 = u1[0, 0], u1[0, 1]
        x2, v2 = u2[0, 0], u2[0, 1]
        
        coupling1 = np.zeros((1, 2))
        coupling2 = np.zeros((1, 2))
        
        coupling1[0, 1] = -epsilon * (v1 - v2)  # -μ(ẋ₁ - ẋ₂) = μ(ẋ₂ - ẋ₁)
        coupling2[0, 1] = -epsilon * (v2 - v1)  # -μ(ẋ₂ - ẋ₁) = μ(ẋ₁ - ẋ₂)
        
        return coupling1, coupling2

