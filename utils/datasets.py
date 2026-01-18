import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

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


def process_trajectory_batch(args):
    """Обрабатывает пачку траекторий и возвращает локальные X, y и статистику"""
    (batch_size, evolution_operator, right_part, 
     variables_ranges, parameters_ranges, n_transient, 
     steps_per_trajectory, fp_threshold, div_threshold,
     secant_plane, secant_plane_derivatives, accuracy, dt, base_seed) = args
    
    rng = np.random.default_rng(base_seed)
    X_local, y_local = []
    stats = {
        "traj_processed": 0,
        "rejected_fixed_points": 0,
        "divergence_trajs_number": 0,
        "accepted_trajectories": 0,
        "exceptions": 0
    }
    
    for _ in range(batch_size):
        # Генерация случайных начальных условий
        x_init = rng.uniform([r[0] for r in variables_ranges], [r[1] for r in variables_ranges])
        p_init = rng.uniform([r[0] for r in parameters_ranges], [r[1] for r in parameters_ranges])
        
        stats["traj_processed"] += 1
        
        # Определение типа аттрактора
        regime = calculate_dynamic_regime(...)
        
        if regime.get("type") == "EP":
            stats["rejected_fixed_points"] += 1
            continue
        
        # Генерация шагов
        x, p = x_init.copy(), p_init.copy()
        try:
            for step in range(steps_per_trajectory):
                x_next = evolution_operator(x, p, dt)
                X_local.append(np.concatenate([x, p]))
                y_local.append(x_next - x)
                x = x_next
                
                if np.any(np.isnan(x)) or np.any(np.abs(x) > div_threshold):
                    stats["divergence_trajs_number"] += 1
                    break
            else:
                stats["accepted_trajectories"] += 1
                
        except Exception:
            stats["exceptions"] += 1
            continue
    
    return np.array(X_local), np.array(y_local), stats

def generate_pairs_dataset_filtered(
    evolution_operator, right_part,
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
    seed=52,
    n_jobs=-1,
    batch_size=10,
    verbose=True
):
    n_var = len(variables_ranges)
    n_param = len(parameters_ranges)
    
    X, y = [], []
    total_stats = {
        "total_trajectories_processed": 0,
        "rejected_fixed_points": 0,
        "divergence_trajs_number": 0,
        "accepted_trajectories": 0,
        "total_samples_generated": 0,
        "target_samples": target_samples
    }
    
    # Прогресс-бар
    pbar = tqdm(total=target_samples, desc="Generating samples", disable=not verbose)
    
    seed_seq = np.random.SeedSequence(seed)
    
    while len(X) < target_samples:
        n_batches = (target_samples - len(X) + batch_size - 1) // batch_size
        
        args_list = [
            (batch_size, evolution_operator, right_part,
             variables_ranges, parameters_ranges, n_transient,
             steps_per_trajectory, fp_threshold, div_threshold,
             secant_plane, secant_plane_derivatives, accuracy, dt,
             seed_seq.spawn(1)[0].entropy)
            for _ in range(n_batches)
        ]
        
        results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(process_trajectory_batch)(args) for args in args_list
        )
        
        for X_batch, y_batch, stats_batch in results:
            if len(X_batch) > 0:
                X.append(X_batch)
                y.append(y_batch)
                pbar.update(len(X_batch))
            
            total_stats["total_trajectories_processed"] += stats_batch["traj_processed"]
            total_stats["rejected_fixed_points"] += stats_batch["rejected_fixed_points"]
            total_stats["divergence_trajs_number"] += stats_batch["divergence_trajs_number"]
            total_stats["accepted_trajectories"] += stats_batch["accepted_trajectories"]
    
    pbar.close()
    
    X = np.concatenate(X)[:target_samples]
    y = np.concatenate(y)[:target_samples]
    total_stats["total_samples_generated"] = len(X)
    
    return X, y, total_stats


