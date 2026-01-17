import numpy as np
from .logger import get_logger
from typing import List, Union, Optional
from joblib import Parallel, delayed
from tqdm import tqdm

from utils.rk4 import rk4_step_vectorized_params

logger = get_logger(__name__)


def pass_transient_process(evolution_operator, state, params, dt, 
                           required_number_of_intersections, secant_plane,
                           fixed_point_threshold=1e-12, max_steps=100_000_000,
                           divergence_threshold=1e5) -> np.ndarray | None:
    logger.debug(f"Начало прохождения переходного процесса: required_intersections={required_number_of_intersections}, max_steps={max_steps}")
    
    number_of_intersections = 0
    step_count = 0
    current_state = np.array(state, dtype=np.float64)
    previous_state = None
    
    while number_of_intersections < required_number_of_intersections:
        step_count += 1
        if step_count > max_steps:
            raise ValueError(f"Достигнуто максимальное количество шагов ({max_steps}). Прерывание процесса.")
        
        previous_state = current_state
        current_state = evolution_operator(current_state, params, dt)
        if np.linalg.norm(current_state - [0]*len(state)) > divergence_threshold:
            return None

        if np.linalg.norm(current_state - previous_state) < fixed_point_threshold:
            return current_state

        S_prev = secant_plane(previous_state, params)
        S_curr = secant_plane(current_state, params)
        
        if S_prev < 0 and S_curr >= 0:
            number_of_intersections += 1
            # logger.debug(f"Пересечение #{number_of_intersections} на шаге {step_count}")
    
    logger.debug(f"Переходный процесс завершен: intersections={number_of_intersections}, steps={step_count}")
    return current_state


def get_attractor_trajectory(evolution_operator, right_part,
                             state, params, dt, 
                             n_transient, n_attractor, 
                             secant_plane, secant_plane_derivatives,
                             accuracy=1e-4, max_steps=100_000_000,
                             fixed_point_threshold=1e-12,
                             divergence_threshold=1e5,
                             contain_only_secants=False):
    
    logger.debug(f"Начало получения траектории аттрактора: n_transient={n_transient}, n_attractor={n_attractor}, accuracy={accuracy}")

    state = pass_transient_process(evolution_operator, state, params,
                                   dt, n_transient, secant_plane, fixed_point_threshold, max_steps, divergence_threshold)

    if state is None: return None
    
    logger.debug(f"Состояние после переходного процесса: {state}")

    attractor_trajectory = []
    number_of_intersections = 0
    previous_state = None
    first_point = None

    while number_of_intersections < n_attractor:
        previous_state = state
        state = evolution_operator(state, params, dt)
        
        if np.linalg.norm(state - previous_state) < fixed_point_threshold:
            logger.debug(f"Обнаружена неподвижная точка (threshold={fixed_point_threshold}). Возврат точки.")
            return [state]

        def S_system(state, params):
                H_val: float = sum(secant_plane_derivatives(state, params) * right_part(state, params))
                dX_dS: np.array(dtype=float) = right_part(state, params) / H_val
                return dX_dS
        
        if not contain_only_secants:
            attractor_trajectory.append(state)

        S_prev = secant_plane(previous_state, params)
        S_curr = secant_plane(state, params)

        if S_prev < 0 and S_curr >= 0:

            sect_point = rk4_step_vectorized_params(S_system, state, params, -S_curr)

            if contain_only_secants:
                attractor_trajectory.append(state)

            if first_point is None:
                first_point = sect_point
                logger.debug(f"Первая точка пересечения: {first_point}")
            else:
                distance = np.linalg.norm(sect_point - first_point)
                logger.debug(f'X={sect_point}, S(X)={secant_plane(sect_point, params)}, d(p1, X)={distance}')
                if distance < accuracy:
                    logger.debug(f"Замкнутая орбита обнаружена (accuracy={accuracy}). Траектория содержит {len(attractor_trajectory)} точек.")
                    return attractor_trajectory
            
            number_of_intersections += 1
            logger.debug(f"Пересечение аттрактора #{number_of_intersections}")

    logger.debug(f"Траектория аттрактора получена: {len(attractor_trajectory)} точек, {number_of_intersections} пересечений")
    return np.array(attractor_trajectory)


def calculate_dynamic_regime(evolution_operator, right_part,
                            state, params, dt,
                            n_transient, n_attractor,
                            secant_plane, secant_plane_derivatives,
                            accuracy=1e-4, max_steps=100_000_000,
                            fixed_point_threshold=1e-12,
                            divergence_threshold=1e5):

    traj = get_attractor_trajectory(
        evolution_operator, right_part,
        state, params, dt,
        n_transient, n_attractor,
        secant_plane, secant_plane_derivatives,
        accuracy, max_steps,
        fixed_point_threshold,
        divergence_threshold,
        contain_only_secants=True
    )

    if traj is None:
        return {"type": "D"}
    
    if len(traj) == 1:
        return {"type": "EP"}
    
    if len(traj) < n_attractor:
        return {"type": "P", "period": len(traj)}
    
    return {"type": "NP"}


def grid_of_amplitude(evolution_operator,
                      state,
                      params: List[Union[np.ndarray, float, int]],
                      dt,
                      n_transient,
                      n_attractor,
                      secant_plane=None,
                      secant_plane_derivatives=None,
                      accuracy: float = 0.0001,
                      max_steps: int = 100_000_000,
                      fixed_point_threshold: float = 1e-12,
                      n_jobs: int = -1,
                      x_param_index: Optional[int] = None,
                      y_param_index: Optional[int] = None,
                      *,
                      model=None,
                      right_part=None,
                      ) -> np.ndarray:
    """
    Параллельно строит 2-D поле амплитуд аттракторов.
    
    Параметры системы могут быть как фиксированными значениями, так и сетками значений.
    Ровно два параметра должны быть сетками (np.ndarray) для построения 2D поля.
    """
    grid_indices = [i for i, p in enumerate(params) if isinstance(p, np.ndarray)]
    
    if len(grid_indices) != 2:
        raise ValueError(
            f"Должно быть ровно 2 параметра-сетки для построения 2D поля, найдено {len(grid_indices)}. "
            f"Остальные параметры должны быть фиксированными значениями (float/int)."
        )
    
    if x_param_index is None:
        x_param_idx = grid_indices[0]
    else:
        if x_param_index not in grid_indices:
            raise ValueError(f"x_param_index={x_param_index} не указывает на параметр-сетку")
        x_param_idx = x_param_index
    
    if y_param_index is None:
        try:
            y_param_idx = next(idx for idx in grid_indices if idx != x_param_idx)
        except StopIteration:
            raise ValueError("Не найден второй параметр-сетки для оси Y")
    else:
        if y_param_index not in grid_indices:
            raise ValueError(f"y_param_index={y_param_index} не указывает на параметр-сетку")
        if y_param_index == x_param_idx:
            raise ValueError("x_param_index и y_param_index должны указывать на разные параметры")
        y_param_idx = y_param_index
    
    x_grid = params[x_param_idx]
    y_grid = params[y_param_idx]
    
    if x_grid.ndim != 1 or y_grid.ndim != 1:
        raise ValueError("Сеточные параметры должны быть одномерными массивами")
    
    Z = np.empty((len(y_grid), len(x_grid)))
    
    def build_params_local(x_val: float, y_val: float) -> List[float]:
        result = []
        for i, p in enumerate(params):
            if i == x_param_idx:
                result.append(float(x_val))
            elif i == y_param_idx:
                result.append(float(y_val))
            else:
                result.append(float(p))
        return result
    
    if model is not None:
        def worker(i_y: int) -> tuple[int, np.ndarray]:
            y_val = y_grid[i_y]
            row = []
            for xi in x_grid:
                # Получаем траекторию
                trajectory = model.simulate(
                    u0=state, 
                    p=build_params_local(xi, y_val),
                    n_steps=n_transient + n_attractor,
                    verbose=False
                )
                
                # Проверяем на расходимость
                if trajectory is None:
                    amplitude = np.inf  # Расходимость → бесконечная амплитуда
                else:
                    # Вычисляем амплитуду для аттрактора
                    attractor_data = trajectory[n_transient:]
                    amplitude = np.linalg.norm(np.ptp(attractor_data, axis=0))
                
                row.append(amplitude)
            
            return i_y, np.array(row)

        desc = "Вычисление сетки Neuromap (по строкам)"
    else:
        # Режим ODE
        if secant_plane is None:
            raise ValueError("secant_plane должен быть указан при использовании evolution_operator")
        
        def worker(i_y: int) -> tuple[int, np.ndarray]:
            y_val = y_grid[i_y]
            row = []
            for xi in x_grid:
                # Получаем траекторию
                trajectory = get_attractor_trajectory(
                    evolution_operator,
                    right_part,
                    state,
                    build_params_local(xi, y_val),
                    dt,
                    n_transient,
                    n_attractor,
                    secant_plane,
                    secant_plane_derivatives,
                    accuracy,
                    max_steps,
                    fixed_point_threshold
                )
                
                # Проверяем на расходимость
                if trajectory is None:
                    amplitude = np.inf  # Расходимость → бесконечная амплитуда
                else:
                    # Вычисляем амплитуду для аттрактора
                    amplitude = np.linalg.norm(np.ptp(trajectory, axis=0))
                
                row.append(amplitude)
            
            return i_y, np.array(row)

        desc = "Вычисление сетки (по строкам)"

    with tqdm(total=len(y_grid), desc=desc) as pbar:
        for i_y, row in Parallel(n_jobs=n_jobs, backend='loky')(
                delayed(worker)(i) for i in range(len(y_grid))):
            Z[i_y] = row
            pbar.update(1)

    return Z


def grid_of_dinamical_regimes(
        evolution_operator,
        state,
        params: List[Union[np.ndarray, float, int]],
        dt,
        n_transient,
        n_attractor,
        secant_plane=None,
        secant_plane_derivatives=None,
        accuracy: float = 0.0001,
        max_steps: int = 100_000_000,
        fixed_point_threshold: float = 1e-12,
        n_jobs: int = -1,
        x_param_index: Optional[int] = None,
        y_param_index: Optional[int] = None,
        *,
        model=None,
        right_part=None,
) -> np.ndarray:
    """
    Параллельно строит 2-D поле динамических режимов аттракторов.
    
    Параметры системы могут быть как фиксированными значениями, так и сетками значений.
    Ровно два параметра должны быть сетками (np.ndarray) для построения 2D поля.
    """
    grid_indices = [i for i, p in enumerate(params) if isinstance(p, np.ndarray)]
    
    if len(grid_indices) != 2:
        raise ValueError(
            f"Должно быть ровно 2 параметра-сетки для построения 2D поля, найдено {len(grid_indices)}. "
            f"Остальные параметры должны быть фиксированными значениями (float/int)."
        )
    
    if x_param_index is None:
        x_param_idx = grid_indices[0]
    else:
        if x_param_index not in grid_indices:
            raise ValueError(f"x_param_index={x_param_index} не указывает на параметр-сетку")
        x_param_idx = x_param_index
    
    if y_param_index is None:
        try:
            y_param_idx = next(idx for idx in grid_indices if idx != x_param_idx)
        except StopIteration:
            raise ValueError("Не найден второй параметр-сетки для оси Y")
    else:
        if y_param_index not in grid_indices:
            raise ValueError(f"y_param_index={y_param_index} не указывает на параметр-сетку")
        if y_param_index == x_param_index:
            raise ValueError("x_param_index и y_param_index должны указывать на разные параметры")
        y_param_idx = y_param_index
    
    x_grid = params[x_param_idx]
    y_grid = params[y_param_idx]
    
    if x_grid.ndim != 1 or y_grid.ndim != 1:
        raise ValueError("Сеточные параметры должны быть одномерными массивами")
    
    # Initialize result grid with strings
    Z = np.empty((len(y_grid), len(x_grid)), dtype=object)
    
    def build_params_local(x_val: float, y_val: float) -> List[float]:
        result = []
        for i, p in enumerate(params):
            if i == x_param_idx:
                result.append(float(x_val))
            elif i == y_param_idx:
                result.append(float(y_val))
            else:
                result.append(float(p))
        return result
    
    # ODE case: use evolution_operator and right_part
    if model is not None:
        raise NotImplementedError("model-based mode not implemented yet")
    else:
        # Validate ODE case requirements
        if secant_plane is None:
            raise ValueError("secant_plane must be provided for ODE case")
        
        def worker(i_y: int) -> tuple[int, np.ndarray]:
            y_val = y_grid[i_y]
            row = []
            for xi in x_grid:
                # Build parameter list for this point
                p_list = build_params_local(xi, y_val)
                
                # Compute regime
                regime = calculate_dynamic_regime(
                    evolution_operator,
                    right_part,
                    state,
                    p_list,
                    dt,
                    n_transient,
                    n_attractor,
                    secant_plane,
                    secant_plane_derivatives,
                    accuracy=accuracy,
                    max_steps=max_steps,
                    fixed_point_threshold=fixed_point_threshold,
                )
                
                # Extract regime type (as string)
                regime_type = regime.get("type", "unknown")
                row.append(regime_type)
            
            return i_y, np.array(row, dtype=object)

        desc = "Вычисление сетки динамических режимов (по строкам)"

    with tqdm(total=len(y_grid), desc=desc) as pbar:
        for i_y, row in Parallel(n_jobs=n_jobs, backend='loky')(
                delayed(worker)(i) for i in range(len(y_grid))):
            Z[i_y] = row
            pbar.update(1)

    return Z

