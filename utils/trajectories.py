import numpy as np
from .logger import get_logger
from typing import List, Union, Optional
from joblib import Parallel, delayed
from tqdm import tqdm

from utils.rk4 import rk4_step_vectorized_params

logger = get_logger(__name__)


def integrate_evolution_operator(
    evolution_operator,
    state,
    params: List[float],
    dt: float,
    n_steps: int,
    divergence_threshold: float,
) -> Optional[np.ndarray]:
    """
    Прямое интегрирование `evolution_operator` (как шаги в Neuromap.simulate):
    траектория длины ``n_steps + 1``: ``u0, u1, …, u_{n_steps}``.
    """
    u = np.asarray(state, dtype=np.float64).copy()
    trajectory = [u.copy()]
    for _ in range(n_steps):
        u = evolution_operator(u, params, dt)
        if np.linalg.norm(u) > divergence_threshold:
            return None
        trajectory.append(u.copy())
    return np.stack(trajectory, axis=0)


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
        # Расходимость может произойти и после прохождения переходного режима.
        if np.linalg.norm(state) > divergence_threshold:
            return None
        
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
                      show_grid_progress: bool = True,
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

    ``show_grid_progress``: tqdm по завершённым строкам сетки (без внутреннего tqdm в ``simulate``).
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

    pbar = tqdm(total=len(y_grid), desc=desc, disable=not show_grid_progress)
    for i_y, row in Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(worker)(i) for i in range(len(y_grid))):
        Z[i_y] = row
        pbar.update(1)
    pbar.close()

    return Z


def grid_of_amplitude_basin(
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
    divergence_threshold: float = 1e5,
    n_jobs: int = -1,
    show_grid_progress: bool = True,
    x_param_index: Optional[int] = None,
    y_param_index: Optional[int] = None,
    *,
    model=None,
    right_part=None,
    ) -> tuple[np.ndarray, np.ndarray]:
    """
    Считает "бассейн амплитуд" на сетке (x,y) с отдельной маской расходимости.

    Для клеток с расходимостью амплитуда возвращается как `np.nan`, а в `divergence_mask=True`.

    Returns:
        (Z, divergence_mask)
        Z: float array shape (len(y), len(x))
        divergence_mask: bool array shape (len(y), len(x))

    ``show_grid_progress``: tqdm по завершённым строкам сетки.
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

    Z = np.empty((len(y_grid), len(x_grid)), dtype=float)
    divergence_mask = np.zeros((len(y_grid), len(x_grid)), dtype=bool)

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
        def worker(i_y: int) -> tuple[int, np.ndarray, np.ndarray]:
            y_val = y_grid[i_y]
            row_amp: List[float] = []
            row_div: List[bool] = []

            for xi in x_grid:
                # У разных реализаций NeuroMap `simulate(...)` может отличаться по аргументам,
                # поэтому здесь делаем совместимый вызов.
                try:
                    trajectory = model.simulate(
                        u0=state,
                        p=build_params_local(xi, y_val),
                        n_steps=n_transient + n_attractor,
                        verbose=False,
                        divergence_threshold=divergence_threshold,
                    )
                except TypeError:
                    trajectory = model.simulate(
                        u0=state,
                        p=build_params_local(xi, y_val),
                        n_steps=n_transient + n_attractor,
                        divergence_threshold=divergence_threshold,
                    )

                if trajectory is None:
                    row_amp.append(np.nan)
                    row_div.append(True)
                else:
                    attractor_data = trajectory[n_transient:]
                    amplitude = np.linalg.norm(np.ptp(attractor_data, axis=0))
                    row_amp.append(float(amplitude))
                    row_div.append(False)

            return i_y, np.array(row_amp, dtype=float), np.array(row_div, dtype=bool)

        desc = "Вычисление сетки Neuromap (амплитуда + расходимость)"
    else:
        if secant_plane is None:
            raise ValueError("secant_plane должен быть указан при использовании evolution_operator")
        if right_part is None:
            raise ValueError("right_part должен быть указан при использовании evolution_operator")
        if secant_plane_derivatives is None:
            raise ValueError("secant_plane_derivatives должен быть указан при использовании evolution_operator")

        def worker(i_y: int) -> tuple[int, np.ndarray, np.ndarray]:
            y_val = y_grid[i_y]
            row_amp: List[float] = []
            row_div: List[bool] = []

            for xi in x_grid:
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
                    fixed_point_threshold,
                    divergence_threshold,
                )

                if trajectory is None:
                    row_amp.append(np.nan)
                    row_div.append(True)
                else:
                    amplitude = np.linalg.norm(np.ptp(trajectory, axis=0))
                    row_amp.append(float(amplitude))
                    row_div.append(False)

            return i_y, np.array(row_amp, dtype=float), np.array(row_div, dtype=bool)

        desc = "Вычисление сетки ODE (амплитуда + расходимость)"

    pbar = tqdm(total=len(y_grid), desc=desc, disable=not show_grid_progress)
    for i_y, row_amp, row_div in Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(worker)(i) for i in range(len(y_grid))
    ):
        Z[i_y] = row_amp
        divergence_mask[i_y] = row_div
        pbar.update(1)
    pbar.close()

    return Z, divergence_mask


def grid_of_amplitude_basin_over_initial_state(
    evolution_operator,
    state,
    params: List[Union[np.ndarray, float, int]],
    dt,
    n_transient,
    n_attractor,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    x_state_index: int,
    y_state_index: int,
    secant_plane=None,
    secant_plane_derivatives=None,
    accuracy: float = 0.0001,
    max_steps: int = 100_000_000,
    fixed_point_threshold: float = 1e-12,
    divergence_threshold: float = 1e5,
    n_jobs: int = -1,
    *,
    model=None,
    right_part=None,
    ode_amplitude_mode: str = "integrated",
    show_grid_progress: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Считает "бассейн амплитуд" на сетке по начальным условиям `u0`.

    Меняются две компоненты начального состояния:
    - компонент с индексом `x_state_index` пробегает `x_grid`
    - компонент с индексом `y_state_index` пробегает `y_grid`

    Параметры системы (`params`) фиксированы для всех точек сетки.

    ``ode_amplitude_mode``:
        - ``"integrated"`` (по умолчанию): ODE и Neuromap считают амплитуду одинаково —
          прямое интегрирование на ``n_transient + n_attractor`` шагов и ``ptp`` на
          участке ``[n_transient:]`` (как в ``model.simulate``).
        - ``"secant"``: старая схема через ``get_attractor_trajectory`` (секущая Пуанкаре);
          тогда нужны ``secant_plane``, ``secant_plane_derivatives``, ``right_part``.

    ``show_grid_progress``: tqdm по завершённым строкам сетки (параллельный расчёт).
        Передать ``False``, чтобы отключить полоску.

    Возвращает:
        (Z, divergence_mask)
        Z: амплитуда, где расходимость -> `np.nan`
        divergence_mask: True там, где траектория разошлась
    """
    if x_state_index == y_state_index:
        raise ValueError("x_state_index и y_state_index должны быть разными")

    x_grid = np.asarray(x_grid, dtype=float)
    y_grid = np.asarray(y_grid, dtype=float)

    if x_grid.ndim != 1 or y_grid.ndim != 1:
        raise ValueError("x_grid и y_grid должны быть одномерными массивами")

    base_state = np.asarray(state, dtype=float).copy()
    p_fixed = [float(p) for p in params]

    Z = np.empty((len(y_grid), len(x_grid)), dtype=float)
    divergence_mask = np.zeros((len(y_grid), len(x_grid)), dtype=bool)

    def build_state_local(x_val: float, y_val: float) -> np.ndarray:
        u0 = base_state.copy()
        u0[x_state_index] = float(x_val)
        u0[y_state_index] = float(y_val)
        return u0

    if model is not None:
        def worker(i_y: int) -> tuple[int, np.ndarray, np.ndarray]:
            y_val = y_grid[i_y]
            row_amp: List[float] = []
            row_div: List[bool] = []

            for xi in x_grid:
                u0_local = build_state_local(xi, y_val)

                try:
                    trajectory = model.simulate(
                        u0=u0_local,
                        p=p_fixed,
                        n_steps=n_transient + n_attractor,
                        verbose=False,
                        divergence_threshold=divergence_threshold,
                    )
                except TypeError:
                    trajectory = model.simulate(
                        u0=u0_local,
                        p=p_fixed,
                        n_steps=n_transient + n_attractor,
                        divergence_threshold=divergence_threshold,
                    )

                if trajectory is None:
                    row_amp.append(np.nan)
                    row_div.append(True)
                else:
                    attractor_data = trajectory[n_transient:]
                    amplitude = np.linalg.norm(np.ptp(attractor_data, axis=0))
                    row_amp.append(float(amplitude))
                    row_div.append(False)

            return i_y, np.array(row_amp, dtype=float), np.array(row_div, dtype=bool)

        desc = "Сетка по u0 (Neuromap): амплитуда + расходимость"

    else:
        if ode_amplitude_mode not in ("integrated", "secant"):
            raise ValueError('ode_amplitude_mode должен быть "integrated" или "secant"')

        if ode_amplitude_mode == "integrated":

            def worker(i_y: int) -> tuple[int, np.ndarray, np.ndarray]:
                y_val = y_grid[i_y]
                row_amp: List[float] = []
                row_div: List[bool] = []
                n_steps = n_transient + n_attractor

                for xi in x_grid:
                    u0_local = build_state_local(xi, y_val)
                    trajectory = integrate_evolution_operator(
                        evolution_operator,
                        u0_local,
                        p_fixed,
                        dt,
                        n_steps,
                        divergence_threshold,
                    )
                    if trajectory is None:
                        row_amp.append(np.nan)
                        row_div.append(True)
                    else:
                        attractor_data = trajectory[n_transient:]
                        amplitude = np.linalg.norm(np.ptp(attractor_data, axis=0))
                        row_amp.append(float(amplitude))
                        row_div.append(False)

                return i_y, np.array(row_amp, dtype=float), np.array(row_div, dtype=bool)

            desc = "Сетка по u0 (ODE, как Neuromap): амплитуда + расходимость"
        else:
            if secant_plane is None:
                raise ValueError("secant_plane должен быть указан при ode_amplitude_mode='secant'")
            if right_part is None:
                raise ValueError("right_part должен быть указан при ode_amplitude_mode='secant'")
            if secant_plane_derivatives is None:
                raise ValueError("secant_plane_derivatives должен быть указан при ode_amplitude_mode='secant'")

            def worker(i_y: int) -> tuple[int, np.ndarray, np.ndarray]:
                y_val = y_grid[i_y]
                row_amp: List[float] = []
                row_div: List[bool] = []

                for xi in x_grid:
                    u0_local = build_state_local(xi, y_val)
                    trajectory = get_attractor_trajectory(
                        evolution_operator,
                        right_part,
                        u0_local,
                        p_fixed,
                        dt,
                        n_transient,
                        n_attractor,
                        secant_plane,
                        secant_plane_derivatives,
                        accuracy,
                        max_steps,
                        fixed_point_threshold,
                        divergence_threshold,
                    )

                    if trajectory is None:
                        row_amp.append(np.nan)
                        row_div.append(True)
                    else:
                        amplitude = np.linalg.norm(np.ptp(trajectory, axis=0))
                        row_amp.append(float(amplitude))
                        row_div.append(False)

                return i_y, np.array(row_amp, dtype=float), np.array(row_div, dtype=bool)

            desc = "Сетка по u0 (ODE, секущая): амплитуда + расходимость"

    pbar = tqdm(
        total=len(y_grid),
        desc=desc,
        disable=not show_grid_progress,
    )
    for i_y, row_amp, row_div in Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(worker)(i) for i in range(len(y_grid))
    ):
        Z[i_y] = row_amp
        divergence_mask[i_y] = row_div
        pbar.update(1)
    pbar.close()

    return Z, divergence_mask


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

