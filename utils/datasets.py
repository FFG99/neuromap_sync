import numpy as np
import multiprocessing as mp
from typing import Callable, Optional, Tuple, Dict, Any
import os
import json
import warnings

from .trajectories import calculate_dynamic_regime


def generate_pairs_dataset_finite(evolution_operator,
                                  variables_ranges,
                                  parameters_ranges,
                                  num_of_traj,
                                  num_in_traj,
                                  dt,
                                  seed=52,
                                  max_attempts_factor=10,
                                  divergence_threshold=1e4,
                                  delta_threshold=None):
    """
    Генерация датасета: X = [u, p], y = Δu.
    Траектории, в которых появляются NaN/Inf или расхождение по норме состояния/приращения,
    отбрасываются полностью и заменяются новыми.

    Args:
        evolution_operator: функция (x, params, dt) -> x_next
        variables_ranges: список [(min, max), ...] для переменных, длина = n_var
        parameters_ranges: список [(min, max), ...] для параметров, длина = n_param
        num_of_traj: количество траекторий
        num_in_traj: число шагов на траектории
        dt: шаг по времени
        seed: зерно ГПСЧ
        max_attempts_factor: множитель для максимального числа попыток
                             (попытки = max_attempts_factor * num_of_traj)
        divergence_threshold: траектория отбрасывается, если max(|x|) или max(|x_next|)
                             превышает это значение (по умолчанию 1e4).
        delta_threshold: траектория отбрасывается, если max(|delta|) по шагу превышает
                         это значение. По умолчанию None (не проверять). Можно задать
                         например 1e3 для защиты от гигантских приращений.

    Returns:
        X: array (N, n_var + n_param) - состояния и параметры
        y: array (N, n_var) - приращения
    """
    rng = np.random.default_rng(seed)
    n_var = len(variables_ranges)
    n_param = len(parameters_ranges)

    X_list = []
    y_list = []
    successful_trajs = 0
    max_attempts = max_attempts_factor * num_of_traj
    attempts = 0

    while successful_trajs < num_of_traj and attempts < max_attempts:
        attempts += 1

        # Генерация начального состояния и параметров
        x0 = rng.uniform(*zip(*variables_ranges), size=n_var)
        p = rng.uniform(*zip(*parameters_ranges), size=n_param)

        x = x0.copy()
        traj_valid = True
        traj_pairs_X = []
        traj_pairs_y = []

        for step in range(num_in_traj):
            x_next = evolution_operator(x, p, dt)

            # Проверка на NaN/Inf
            if not np.all(np.isfinite(x_next)):
                traj_valid = False
                break

            # Порог расходимости по состоянию
            if np.max(np.abs(x_next)) > divergence_threshold or np.max(np.abs(x)) > divergence_threshold:
                traj_valid = False
                break

            delta = x_next - x
            if not np.all(np.isfinite(delta)):
                traj_valid = False
                break
            if delta_threshold is not None and np.max(np.abs(delta)) > delta_threshold:
                traj_valid = False
                break

            traj_pairs_X.append(np.concatenate([x, p]))
            traj_pairs_y.append(delta)
            x = x_next

        if traj_valid:
            X_list.extend(traj_pairs_X)
            y_list.extend(traj_pairs_y)
            successful_trajs += 1

    if successful_trajs < num_of_traj:
        raise RuntimeError(
            f"Не удалось сгенерировать {num_of_traj} валидных траекторий "
            f"за {max_attempts} попыток. Собрано только {successful_trajs}."
        )

    return np.array(X_list), np.array(y_list)


def generate_pairs_dataset(evolution_operator,
                           variables_ranges,
                           parameters_ranges,
                           num_of_traj,
                           num_in_traj,
                           dt,
                           seed=52):
    """
    Генерация датасета: X = [u, p], y = Δu
    
    Args:
        evolution_operator: функция (x, params, dt) -> x_next
        variables_ranges: список [(min, max), ...] для переменных, длина = n_var
        parameters_ranges: список [(min, max), ...] для параметров, длина = n_param
        num_of_traj: количество траекторий
        num_in_traj: число шагов на траектории
        dt: шаг по времени
    Returns:
        X: array (N, n_var + n_param) - состояния и параметры
        y: array (N, n_var) - приращения
    """
    rng = np.random.default_rng(seed)
    
    n_var = len(variables_ranges)
    n_param = len(parameters_ranges)
    
    variables  = rng.uniform(*zip(*variables_ranges),  size=(num_of_traj, n_var))
    parameters = rng.uniform(*zip(*parameters_ranges), size=(num_of_traj, n_param))
    
    X, y = [], []
    for traj in range(num_of_traj):
        x = variables[traj].copy()
        p = parameters[traj].copy()
        for _ in range(num_in_traj):
            x_next = evolution_operator(x, p, dt)
            delta = x_next - x
            X.append(np.concatenate([x, p]))
            y.append(delta)
            x = x_next
    
    return np.array(X), np.array(y)


class DynamicSystemDatasetGenerator:
    """
    Генератор датасета пар (состояние, параметры) -> приращение для динамических систем
    с фильтрацией по типу аттрактора.
    """
    def __init__(self,
                 evolution_operator: Callable,
                 right_part: Callable,
                 variables_ranges: list,
                 parameters_ranges: list,
                 n_transient: int = 200,
                 steps_per_trajectory: int = 5,
                 fp_threshold: float = 1e-12,
                 div_threshold: float = 1e5,
                 secant_plane: Optional[Callable] = None,
                 secant_plane_derivatives: Optional[Callable] = None,
                 accuracy: float = 1e-3,
                 dt: float = 0.01,
                 seed: int = 52):
        """
        Параметры:
            evolution_operator: Функция эволюции системы: (state, params, dt) -> next_state
            right_part: Правая часть ДУ для определения типа аттрактора
            variables_ranges: Диапазоны переменных фазового пространства: [(min, max), ...]
            parameters_ranges: Диапазоны параметров системы: [(min, max), ...]
            n_transient: Число шагов для трансиентного процесса
            steps_per_trajectory: Сколько первых шагов брать с каждой траектории
            fp_threshold: Порог для определения неподвижной точки
            div_threshold: Порог для определения расходимости
            secant_plane: Функция секущей плоскости для определения типа аттрактора
            secant_plane_derivatives: Функция производных секущей плоскости
            accuracy: Точность для определения типа аттрактора
            dt: Шаг интегрирования
            seed: Seed для генератора случайных чисел
        """
        self.evolution_operator = evolution_operator
        self.right_part = right_part
        self.variables_ranges = variables_ranges
        self.parameters_ranges = parameters_ranges
        self.n_transient = n_transient
        self.steps_per_trajectory = steps_per_trajectory
        self.fp_threshold = fp_threshold
        self.div_threshold = div_threshold
        self.secant_plane = secant_plane
        self.secant_plane_derivatives = secant_plane_derivatives
        self.accuracy = accuracy
        self.dt = dt
        self.seed = seed
        
        self.rng = np.random.default_rng(seed)
        self.n_var = len(variables_ranges)
        self.n_param = len(parameters_ranges)
        
        self.X: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self.info: Optional[Dict[str, Any]] = None

    def generate(self, target_samples: int, resume: bool = False, 
        n_jobs: int = -1) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Генерация датасета.
        
        Args:
            target_samples: Целевое количество пар (X, y)
            resume: Если True, продолжает генерацию существующего датасета
            n_jobs: Количество параллельных задач (-1 = все ядра)
            
        Returns:
            X: Массив состояний и параметров (N, n_var + n_param)
            y: Массив приращений (N, n_var)
            info: Статистика генерации
        """
        import concurrent.futures
        
        if n_jobs == -1:
            n_jobs = mp.cpu_count()
        
        if resume and self.X is not None:
            X_list = self.X.tolist()
            y_list = self.y.tolist()
            start_count = len(X_list)
            info = self.info.copy() if self.info else self._init_info()
        else:
            X_list, y_list = [], []
            start_count = 0
            info = self._init_info()
        
        print(f"Начало генерации. Цель: {target_samples} образцов. "
            f"Начальное количество: {start_count}. Параллелизм: {n_jobs} потоков")
        
        # Пулы для параллельного вычисления режимов
        futures_pool = []
        pending_points = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:
            # Инициализируем первый батч задач
            batch_size = min(n_jobs * 2, 100)
            for _ in range(batch_size):
                if len(X_list) >= target_samples:
                    break
                    
                x_init, p_init = self._generate_random_point()
                future = executor.submit(self._calculate_regime, x_init, p_init)
                futures_pool.append((future, x_init, p_init))
            
            while len(X_list) < target_samples:
                if not futures_pool:
                    break
                    
                # Ждем завершения следующей задачи
                done, _ = concurrent.futures.wait(
                    [f[0] for f in futures_pool], 
                    return_when=concurrent.futures.FIRST_COMPLETED
                )
                
                # Обрабатываем завершенные задачи
                new_futures_pool = []
                for future, x_init, p_init in futures_pool:
                    if future in done:
                        info['total_trajectories_processed'] += 1
                        
                        try:
                            regime = future.result()
                            
                            # Пропуск неподвижных точек
                            if regime.get("type") == "EP":
                                info['rejected_fixed_points'] += 1
                            else:
                                # Генерация шагов траектории
                                if self._generate_trajectory_steps(x_init, p_init, X_list, y_list, info):
                                    info['accepted_trajectories'] += 1
                        except Exception as e:
                            info['errors'] = info.get('errors', 0) + 1
                            if info['errors'] % 10 == 0:  # Логируем каждую 10-ю ошибку
                                print(f"Ошибка при обработке траектории: {e}")
                    else:
                        new_futures_pool.append((future, x_init, p_init))
                
                # Добавляем новые задачи для поддержания пула
                while len(new_futures_pool) < batch_size and len(X_list) < target_samples:
                    x_init, p_init = self._generate_random_point()
                    future = executor.submit(self._calculate_regime, x_init, p_init)
                    new_futures_pool.append((future, x_init, p_init))
                
                futures_pool = new_futures_pool
                
                # Прогресс каждые 100 траекторий
                if info['total_trajectories_processed'] % 100 == 0:
                    print(f"Сгенерировано {len(X_list)}/{target_samples} образцов "
                        f"(обработано {info['total_trajectories_processed']} траекторий)")
        
        # Обрезаем до target_samples
        self.X = np.array(X_list[:target_samples])
        self.y = np.array(y_list[:target_samples])
        
        # Обновляем статистику
        info['total_samples_generated'] = len(self.X)
        info['target_samples'] = target_samples
        info['n_jobs_used'] = n_jobs
        self.info = info
        
        print(f"Генерация завершена. Всего образцов: {len(self.X)}")
        
        return self.X, self.y, self.info

    def _generate_random_point(self) -> Tuple[np.ndarray, np.ndarray]:
        """Генерация случайной начальной точки."""
        x_init = self.rng.uniform(
            [r[0] for r in self.variables_ranges],
            [r[1] for r in self.variables_ranges]
        )
        p_init = self.rng.uniform(
            [r[0] for r in self.parameters_ranges],
            [r[1] for r in self.parameters_ranges]
        )
        return x_init, p_init

    def _generate_trajectory_steps(self, x_init: np.ndarray, p_init: np.ndarray,
                                X_list: list, y_list: list, info: Dict) -> bool:
        """Генерация шагов траектории и добавление в датасет."""
        x = x_init.copy()
        p = p_init.copy()
        
        samples_generated = 0
        samples_rejected = 0
        
        try:
            for step in range(self.steps_per_trajectory):
                x_next = self.evolution_operator(x, p, self.dt)
                delta = x_next - x
                
                if (np.any(np.isinf(x_next)) or np.any(np.isnan(x_next)) or 
                    np.any(np.isinf(delta)) or np.any(np.isnan(delta))):
                    samples_rejected += 1
                    x = x_next
                    continue
                
                X_list.append(np.concatenate([x, p]))
                y_list.append(delta)
                samples_generated += 1
                
                x = x_next
                                    
        except Exception as e:
            # Обновляем статистику ошибок
            info['trajectory_errors'] = info.get('trajectory_errors', 0) + 1
            return False
        
        # Обновляем статистику
        if 'samples_rejected' not in info:
            info['samples_rejected'] = 0
            info['samples_generated_per_trajectory'] = []
        
        info['samples_rejected'] += samples_rejected
        info['samples_generated_per_trajectory'].append(samples_generated)
        
        # Возвращаем True, если сгенерирован хотя бы один валидный шаг
        return samples_generated > 0
    
    def _init_info(self) -> Dict[str, Any]:
        """Инициализация словаря статистики."""
        return {
            'total_trajectories_processed': 0,
            'rejected_fixed_points': 0,
            'divergence_trajs_number': 0,
            'accepted_trajectories': 0,
            'total_samples_generated': 0,
            'target_samples': 0
        }
    
    def _calculate_regime(self, x_init: np.ndarray, p_init: np.ndarray) -> Dict[str, Any]:
        """Определение типа аттрактора."""
        return calculate_dynamic_regime(
            evolution_operator=self.evolution_operator,
            right_part=self.right_part,
            state=x_init,
            params=p_init,
            dt=self.dt,
            n_transient=self.n_transient,
            n_attractor=10,
            secant_plane=self.secant_plane,
            secant_plane_derivatives=self.secant_plane_derivatives,
            accuracy=self.accuracy,
            max_steps=100_000,
            fixed_point_threshold=self.fp_threshold,
            divergence_threshold=self.div_threshold
        )
        
    def save(self, filepath: str, overwrite: bool = False) -> None:
        """
        Сохранение датасета и конфигурации в файл.
        
        Args:
            filepath: Путь к файлу (.npz)
            overwrite: Разрешить перезапись существующего файла
        """
        if os.path.exists(filepath) and not overwrite:
            raise FileExistsError(f"Файл {filepath} уже существует. Используйте overwrite=True")
        
        if self.X is None or self.y is None or self.info is None:
            raise ValueError("Нет данных для сохранения. Сначала запустите generate()")
        
        # Создаем директорию, если она не существует
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            print(f"Создана директория: {directory}")
        
        # Сохраняем данные и конфигурацию
        np.savez_compressed(
            filepath,
            X=self.X,
            y=self.y,
            info=np.array(json.dumps(self.info)),  # Сериализуем dict в строку
            # Сохраняем конфигурацию (без функций secant_plane и secant_plane_derivatives)
            config=np.array(json.dumps({
                'variables_ranges': self.variables_ranges,
                'parameters_ranges': self.parameters_ranges,
                'n_transient': self.n_transient,
                'steps_per_trajectory': self.steps_per_trajectory,
                'fp_threshold': self.fp_threshold,
                'div_threshold': self.div_threshold,
                'accuracy': self.accuracy,
                'dt': self.dt,
                'seed': self.seed,
                'n_var': self.n_var,
                'n_param': self.n_param,
                # Сохраняем флаги наличия функций (для информации)
                'has_secant_plane': self.secant_plane is not None,
                'has_secant_plane_derivatives': self.secant_plane_derivatives is not None,
            }))
        )
        print(f"Датасет сохранен в {filepath}")
    
    @classmethod
    def load(cls, filepath: str, 
             evolution_operator: Callable, 
             right_part: Callable,
             secant_plane: Optional[Callable] = None,
             secant_plane_derivatives: Optional[Callable] = None) -> 'DynamicSystemDatasetGenerator':
        """
        Загрузка датасета и конфигурации из файла.
        
        Args:
            filepath: Путь к файлу (.npz)
            evolution_operator: Функция эволюции (не сохраняется в файле)
            right_part: Правая часть ДУ (не сохраняется в файле)
            secant_plane: Функция секущей плоскости (не сохраняется в файле)
            secant_plane_derivatives: Функция производных секущей плоскости (не сохраняется в файле)
            
        Returns:
            Экземпляр DynamicSystemDatasetGenerator с загруженными данными
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Файл {filepath} не найден")
        
        data = np.load(filepath, allow_pickle=True)
        
        # Восстанавливаем конфигурацию
        config = json.loads(str(data['config']))
        
        # Создаем экземпляр
        instance = cls(
            evolution_operator=evolution_operator,
            right_part=right_part,
            variables_ranges=config['variables_ranges'],
            parameters_ranges=config['parameters_ranges'],
            n_transient=config['n_transient'],
            steps_per_trajectory=config['steps_per_trajectory'],
            fp_threshold=config['fp_threshold'],
            div_threshold=config['div_threshold'],
            secant_plane=secant_plane,
            secant_plane_derivatives=secant_plane_derivatives,
            accuracy=config['accuracy'],
            dt=config['dt'],
            seed=config['seed']
        )
        
        # Восстанавливаем данные
        instance.X = data['X']
        instance.y = data['y']
        instance.info = json.loads(str(data['info']))
        
        # Предупреждение, если в сохраненных данных были функции, но при загрузке не переданы
        if config.get('has_secant_plane', False) and secant_plane is None:
            warnings.warn("В сохраненном датасете использовалась secant_plane, но при загрузке передана None.")
        if config.get('has_secant_plane_derivatives', False) and secant_plane_derivatives is None:
            warnings.warn("В сохраненном датасете использовалась secant_plane_derivatives, но при загрузке передана None.")
        
        print(f"Датасет загружен из {filepath}. Образцов: {len(instance.X)}")
        
        return instance
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Возвращает сводку по конфигурации."""
        return {
            'phase_space_dim': self.n_var,
            'parameter_dim': self.n_param,
            'variables_ranges': self.variables_ranges,
            'parameters_ranges': self.parameters_ranges,
            'steps_per_trajectory': self.steps_per_trajectory,
            'transient_steps': self.n_transient,
            'integration_dt': self.dt,
            'seed': self.seed,
            'has_secant_plane': self.secant_plane is not None,
            'has_secant_plane_derivatives': self.secant_plane_derivatives is not None
        }


def generate_sequence_dataset(
    evolution_operator, variables_ranges, parameters_ranges,
    num_of_traj=200_000, seq_len=10, dt=0.01, seed=None
):
    """
    Генерируем список траекторий длиной seq_len+1.
    Возвращаем X: (N, seq_len, n_var + n_param), y: (N, seq_len, n_var)
    где y = u_{t+1} - u_t (приращения).
    """
    rng = np.random.default_rng(seed)
    X_list, y_list = [], []

    for _ in range(num_of_traj):
        # случайный стартовый u и p
        u0 = rng.uniform(*zip(*variables_ranges))
        p  = rng.uniform(*zip(*parameters_ranges))

        # симулируем seq_len+1 шагов
        us = [u0]
        for _ in range(seq_len):
            u_next = evolution_operator(us[-1], p, dt=dt)
            us.append(u_next)

        us = np.array(us)                     # shape (seq_len+1, n_var)
        du  = us[1:] - us[:-1]                # shape (seq_len, n_var)

        X_seq = np.concatenate([us[:-1], np.tile(p, (seq_len, 1))], axis=1)
        # X_seq shape: (seq_len, n_var + n_param)

        X_list.append(X_seq)
        y_list.append(du)                     # shape (seq_len, n_var)

    X = np.stack(X_list, axis=0)  # (N, seq_len, n_var + n_param)
    y = np.stack(y_list, axis=0)  # (N, seq_len, n_var)
    return X, y
