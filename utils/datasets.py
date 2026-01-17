import numpy as np

from .trajectories import calculate_dynamic_regime


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


def generate_pairs_dataset_filtered(evolution_operator, right_part,
                                    variables_ranges,
                                    parameters_ranges,
                                    target_samples,
                                    n_transient=200,
                                    steps_per_trajectory=5,
                                    fp_threshold=1e-12,
                                    div_threshold=1e5,
                                    secant_plane=None,
                                    secant_plane_derivatives=None,
                                    accuracy=1e-3,
                                    dt=0.01,
                                    seed=52):
    """
    Генерация датасета с фильтрацией по типу аттрактора.
    
    Генерирует пары (X, y), где X = [state, parameters], а y = Δstate,
    но только для траекторий, которые НЕ приводят к неподвижной точке 
    (equilibrium point). Для каждой подходящей траектории берутся 
    первые `steps_per_trajectory` шагов.
    
    Args:
        evolution_operator: callable
            Функция эволюции системы: (state, params, dt) -> next_state
        right_part: callable
            Правая часть дифференциального уравнения для определения типа аттрактора
        variables_ranges: list of tuples
            Диапазоны для переменных фазового пространства: [(min, max), ...]
        parameters_ranges: list of tuples
            Диапазоны для параметров системы: [(min, max), ...]
        target_samples: int
            Целевое количество пар (X, y) в датасете
        n_transient: int, optional
            Число шагов для трансиентного процесса при определении типа аттрактора
        steps_per_trajectory: int, optional
            Сколько первых шагов брать с каждой траектории (по умолчанию 5)
        fp_threshold: float, optional
            Порог для определения неподвижной точки
        div_threshold: float, optional
            Порог для определения расходимости траектории
        secant_plane: ndarray or None, optional
            Секущая плоскость для определения типа аттрактора
        secant_plane_derivatives: ndarray or None, optional
            Производные секущей плоскости
        accuracy: float, optional
            Точность для определения типа аттрактора
        dt: float, optional
            Шаг интегрирования по времени
        seed: int, optional
            Seed для генератора случайных чисел
            
    Returns:
        X: ndarray
            Массив состояний и параметров, shape (N, n_var + n_param)
        y: ndarray
            Массив приращений, shape (N, n_var)
        info: dict
            Статистика генерации:
            - 'total_trajectories_processed': общее число обработанных траекторий
            - 'rejected_fixed_points': число отброшенных неподвижных точек
            - 'divergence_trajs_number': число расходящихся траекторий
            - 'accepted_trajectories': число принятых траекторий
            - 'total_samples_generated': фактическое число сгенерированных образцов
            - 'target_samples': целевое число образцов
    """
    rng = np.random.default_rng(seed)
    
    n_var = len(variables_ranges)
    n_param = len(parameters_ranges)
    
    X, y = [], []
    traj_processed = 0
    traj_rejected_ep = 0
    divergence_trajs_number = 0

    while len(X) < target_samples:
        # Случайные начальные условия и параметры
        x_init = rng.uniform(
            [r[0] for r in variables_ranges],
            [r[1] for r in variables_ranges]
        )
        p_init = rng.uniform(
            [r[0] for r in parameters_ranges],
            [r[1] for r in parameters_ranges]
        )
        
        traj_processed += 1
        
        # Определение типа аттрактора
        regime = calculate_dynamic_regime(
            evolution_operator=evolution_operator,
            right_part=right_part,
            state=x_init,
            params=p_init,
            dt=dt,
            n_transient=n_transient,
            n_attractor=10,
            secant_plane=secant_plane,
            secant_plane_derivatives=secant_plane_derivatives,
            accuracy=accuracy,
            max_steps=100_000,
            fixed_point_threshold=fp_threshold,
            divergence_threshold=div_threshold
        )
        
        # Пропуск неподвижных точек
        if regime.get("type") == "EP":
            traj_rejected_ep += 1
            continue

        # Генерация первых steps_per_trajectory шагов
        x = x_init.copy()
        p = p_init.copy()
        
        try:
            for step in range(steps_per_trajectory):
                x_next = evolution_operator(x, p, dt)
                delta = x_next - x
                
                X.append(np.concatenate([x, p]))
                y.append(delta)
                
                x = x_next
                
                # Проверка на расходимость
                if np.any(np.isnan(x)) or np.any(np.abs(x) > div_threshold):
                    divergence_trajs_number += 1
                    break
                    
        except Exception:
            continue
        
        if traj_processed % 100 == 0:
            print(f"Generated {len(X)}/{target_samples} samples "
                  f"(processed {traj_processed} trajectories)")
    
    X = np.array(X[:target_samples])
    y = np.array(y[:target_samples])
    
    info = {
        "total_trajectories_processed": traj_processed,
        "rejected_fixed_points": traj_rejected_ep,
        "divergence_trajs_number": divergence_trajs_number,
        "accepted_trajectories": traj_processed - traj_rejected_ep,
        "total_samples_generated": len(X),
        "target_samples": target_samples
    }
    
    return X, y, info
