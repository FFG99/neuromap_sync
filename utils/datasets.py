import numpy as np


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
