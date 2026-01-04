import numpy as np
from .logger import get_logger
from typing import List
from joblib import Parallel, delayed
from tqdm import tqdm

from utils.rk4 import rk4_step_vectorized_params

logger = get_logger(__name__)


def pass_transient_process(evolution_operator, state, params, dt, 
                           required_number_of_intersections, secant_plane,
                           fixed_point_threshold=1e-12, max_steps=100_000_000) -> np.ndarray | None:
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
                             fixed_point_threshold=1e-12):
    
    logger.debug(f"Начало получения траектории аттрактора: n_transient={n_transient}, n_attractor={n_attractor}, accuracy={accuracy}")

    state = pass_transient_process(evolution_operator, state, params,
                                   dt, n_transient, secant_plane, fixed_point_threshold, max_steps)
    
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
        
        attractor_trajectory.append(state)
        S_prev = secant_plane(previous_state, params)
        S_curr = secant_plane(state, params)

        if S_prev < 0 and S_curr >= 0:

            logger.debug(f'S_system={S_system}, state={state}, params={params}, -S_curr={-S_curr}')
            sect_point = rk4_step_vectorized_params(S_system, state, params, -S_curr)

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


def grid_of_amplitude(evolution_operator,
                      state,
                      params: List[np.ndarray],
                      dt,
                      n_transient,
                      n_attractor,
                      secant_plane=None,
                      accuracy: float = 0.0001,
                      max_steps: int = 100_000_000,
                      fixed_point_threshold: float = 1e-12,
                      n_jobs: int = -1,
                      *,
                      model=None) -> np.ndarray:
    """
    Параллельно строит 2-D поле амплитуд аттракторов.
    state = [x_grid, y_grid] – список из двух np.ndarray (линейки).
    Возвращает Z[i, j] = ‖range(attractor([x[j], y[i]]))‖
    
    Args:
        evolution_operator: функция эволюции системы (используется если model=None)
        state: начальное состояние системы
        params: список из двух сеток параметров [x_grid, y_grid]
        dt: шаг времени
        n_transient: количество пересечений переходного процесса (для ODE) или шагов (для model)
        n_attractor: количество пересечений аттрактора (для ODE) или шагов (для model)
        secant_plane: функция секущей плоскости (используется только для ODE)
        accuracy: точность для обнаружения замкнутой орбиты (только для ODE)
        max_steps: максимальное количество шагов (только для ODE)
        fixed_point_threshold: порог для обнаружения неподвижной точки (только для ODE)
        n_jobs: количество параллельных процессов
        model: модель NeuroMapFixed или NeuroMapOriginal (если указана, используется вместо evolution_operator)
    """
    x_grid, y_grid = params

    Z = np.empty((len(y_grid), len(x_grid)))

    if model is not None:
        # Режим нейромэпа
        def worker(i_y: int) -> tuple[int, np.ndarray]:
            y_val = y_grid[i_y]
            row = np.array([
                np.linalg.norm(np.ptp(
                    model.simulate(u0=state, p=[xi, y_val], n_steps=n_transient + n_attractor,
                        verbose=False)[n_transient:],
                    axis=0))
                for xi in x_grid
            ])
            return i_y, row

        desc = "Вычисление сетки Neuromap (по строкам)"
    else:
        # Режим ODE
        if secant_plane is None:
            raise ValueError("secant_plane должен быть указан при использовании evolution_operator")
        
        def worker(i_y: int) -> tuple[int, np.ndarray]:
            y_val = y_grid[i_y]
            row = np.array([
                np.linalg.norm(np.ptp(get_attractor_trajectory(
                    evolution_operator,
                    state,
                    [xi, y_val],
                    dt,
                    n_transient,
                    n_attractor,
                    secant_plane,
                    accuracy,
                    max_steps,
                    fixed_point_threshold), axis=0))
                for xi in x_grid
            ])
            return i_y, row

        desc = "Вычисление сетки (по строкам)"

    with tqdm(total=len(y_grid), desc=desc) as pbar:
        for i_y, row in Parallel(n_jobs=n_jobs, backend='loky')(
                delayed(worker)(i) for i in range(len(y_grid))):
            Z[i_y] = row
            pbar.update(1)

    return Z
