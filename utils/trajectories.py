import warnings
from contextlib import contextmanager

import numpy as np
from .logger import get_logger
from typing import Any, Dict, List, Optional, Union
from joblib import Parallel, delayed
from tqdm import tqdm

from utils.rk4 import rk4_step_vectorized_params

logger = get_logger(__name__)


@contextmanager
def _suppress_loky_grid_worker_warning():
    """
    loky может выдавать UserWarning при ``return_as='generator'`` и перезапуске воркеров,
    хотя расчёт завершается нормально (см. обсуждения joblib/loky). Подавляем только
    это сообщение на время итерации по результатам ``Parallel``.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"A worker stopped while some jobs were given to the executor",
            category=UserWarning,
        )
        yield


def _parallel_for_grid_jobs(
    n_jobs: int,
    *,
    parallel_extra: Optional[Dict[str, Any]] = None,
) -> Parallel:
    """
    ``joblib.Parallel`` (loky) для сеточных циклов: ``return_as='generator'``.

    По умолчанию **не** задаём ``max_tasks_per_child``: принудительный перезапуск воркеров
    в loky иногда даёт тот же warning «A worker stopped...». Если нужно ограничить память,
    передайте ``parallel_extra={"max_tasks_per_child": 8}`` (или ``1``) и при необходимости
    уменьшите ``n_jobs``. Долгие задачи: ``parallel_extra={"timeout": 86400}``.
    """
    kw: Dict[str, Any] = dict(
        n_jobs=n_jobs,
        backend="loky",
        return_as="generator",
    )
    if parallel_extra:
        kw.update(parallel_extra)
    if kw.get("max_tasks_per_child") is None:
        kw.pop("max_tasks_per_child", None)
    return Parallel(**kw)


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


def full_trajectory_ptp_norm(
    evolution_operator,
    state,
    params,
    dt: float,
    n_burn_in_steps: int,
    n_record_steps: int,
    divergence_threshold: float,
) -> Optional[float]:
    """
    Амплитуда по **полной** дискретной траектории (шаги ``evolution_operator``).

    Интегрирует ``n_burn_in_steps + n_record_steps`` шагов; при выходе за
    ``divergence_threshold`` возвращает ``None``. Иначе
    ``||\\mathrm{ptp}(U)\\||_2``, где ``U`` — состояния на отрезке после выгорания
    трансиента (хвост длины ``n_record_steps + 1`` точек, начиная с шага
    ``n_burn_in_steps``).
    """
    n_total = int(n_burn_in_steps) + int(n_record_steps)
    if n_total < 0:
        raise ValueError("n_burn_in_steps + n_record_steps must be non-negative")
    if n_record_steps < 1:
        raise ValueError("n_record_steps must be >= 1 to form a span over time")
    traj = integrate_evolution_operator(
        evolution_operator,
        state,
        params,
        dt,
        n_total,
        divergence_threshold,
    )
    if traj is None:
        return None
    tail = traj[int(n_burn_in_steps) :]
    return float(np.linalg.norm(np.ptp(tail, axis=0)))


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
                      parallel_extra: Optional[Dict[str, Any]] = None,
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

    ``parallel_extra``: доп. аргументы для ``joblib.Parallel`` (см. ``_parallel_for_grid_jobs``).
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
    with _suppress_loky_grid_worker_warning():
        for i_y, row in _parallel_for_grid_jobs(n_jobs, parallel_extra=parallel_extra)(
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
    parallel_extra: Optional[Dict[str, Any]] = None,
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

    ``parallel_extra``: доп. аргументы для ``joblib.Parallel`` (см. ``_parallel_for_grid_jobs``).
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
    with _suppress_loky_grid_worker_warning():
        for i_y, row_amp, row_div in _parallel_for_grid_jobs(n_jobs, parallel_extra=parallel_extra)(
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
    parallel_extra: Optional[Dict[str, Any]] = None,
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

    ``parallel_extra``: доп. аргументы для ``joblib.Parallel`` (см. ``_parallel_for_grid_jobs``).

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
    with _suppress_loky_grid_worker_warning():
        for i_y, row_amp, row_div in _parallel_for_grid_jobs(n_jobs, parallel_extra=parallel_extra)(
            delayed(worker)(i) for i in range(len(y_grid))
        ):
            Z[i_y] = row_amp
            divergence_mask[i_y] = row_div
            pbar.update(1)
    pbar.close()

    return Z, divergence_mask


def grid_of_fixed_point_probability_over_params(
    evolution_operator,
    state,
    params: List[Union[np.ndarray, float, int]],
    dt: float,
    n_steps: int,
    x_init_grid: np.ndarray,
    y_init_grid: np.ndarray,
    x_state_index: int,
    y_state_index: int,
    fixed_point_threshold: float = 1e-10,
    divergence_threshold: float = 1e5,
    n_jobs: int = -1,
    show_grid_progress: bool = True,
    parallel_extra: Optional[Dict[str, Any]] = None,
    x_param_index: Optional[int] = None,
    y_param_index: Optional[int] = None,
    *,
    model=None,
) -> np.ndarray:
    """
    Строит 2D-карту вероятности попадания в неподвижную точку по двум параметрам.

    Для каждой пары параметров запускается набор траекторий из сетки начальных условий
    ``(x_init_grid, y_init_grid)``. Вероятность = доля траекторий, у которых выполнено
    условие сходимости ``||u_t - u_(t-1)|| < fixed_point_threshold``.

    Важно: распараллеливание выполняется только по сетке параметров (по строкам y),
    внутри каждой точки параметров расчёт по начальным условиям идёт последовательно.

    Режимы:
    - ``model is not None``: используется ``model.simulate(...)``.
    - ``model is None``: используется ``evolution_operator`` как шаговая функция.
    """
    if x_state_index == y_state_index:
        raise ValueError("x_state_index и y_state_index должны быть разными")
    if n_steps < 2:
        raise ValueError("n_steps должен быть >= 2, чтобы проверить критерий неподвижной точки")

    grid_indices = [i for i, p in enumerate(params) if isinstance(p, np.ndarray)]
    if len(grid_indices) != 2:
        raise ValueError(
            f"Должно быть ровно 2 параметра-сетки, найдено {len(grid_indices)}. "
            "Остальные параметры должны быть фиксированными (float/int)."
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

    x_param_grid = np.asarray(params[x_param_idx], dtype=float)
    y_param_grid = np.asarray(params[y_param_idx], dtype=float)
    if x_param_grid.ndim != 1 or y_param_grid.ndim != 1:
        raise ValueError("Параметры-сетки должны быть одномерными массивами")

    x_init_grid = np.asarray(x_init_grid, dtype=float)
    y_init_grid = np.asarray(y_init_grid, dtype=float)
    if x_init_grid.ndim != 1 or y_init_grid.ndim != 1:
        raise ValueError("x_init_grid и y_init_grid должны быть одномерными массивами")

    base_state = np.asarray(state, dtype=float).copy()
    total_initial_conditions = int(len(x_init_grid) * len(y_init_grid))
    if total_initial_conditions == 0:
        raise ValueError("Сетка начальных условий не должна быть пустой")

    P = np.empty((len(y_param_grid), len(x_param_grid)), dtype=float)

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

    def converges_to_fixed_point(u0: np.ndarray, p_local: List[float]) -> bool:
        if model is not None:
            try:
                trajectory = model.simulate(
                    u0=u0,
                    p=p_local,
                    n_steps=n_steps,
                    verbose=False,
                    divergence_threshold=divergence_threshold,
                )
            except TypeError:
                trajectory = model.simulate(
                    u0=u0,
                    p=p_local,
                    n_steps=n_steps,
                    divergence_threshold=divergence_threshold,
                )

            if trajectory is None or len(trajectory) < 2:
                return False

            last = np.asarray(trajectory[-1], dtype=float)
            prev = np.asarray(trajectory[-2], dtype=float)
            return bool(np.linalg.norm(last - prev) < fixed_point_threshold)

        if evolution_operator is None:
            raise ValueError("Передайте либо model, либо evolution_operator")

        u_prev = np.asarray(u0, dtype=float).copy()
        for _ in range(n_steps):
            u_curr = evolution_operator(u_prev, p_local, dt)
            if np.linalg.norm(u_curr) > divergence_threshold:
                return False
            if np.linalg.norm(u_curr - u_prev) < fixed_point_threshold:
                return True
            u_prev = u_curr
        return False

    def worker(i_y: int) -> tuple[int, np.ndarray]:
        py = y_param_grid[i_y]
        row_probs: List[float] = []
        for px in x_param_grid:
            p_local = build_params_local(px, py)
            fixed_count = 0

            for y0 in y_init_grid:
                for x0 in x_init_grid:
                    u0 = base_state.copy()
                    u0[x_state_index] = float(x0)
                    u0[y_state_index] = float(y0)
                    if converges_to_fixed_point(u0, p_local):
                        fixed_count += 1

            row_probs.append(fixed_count / total_initial_conditions)

        return i_y, np.array(row_probs, dtype=float)

    pbar = tqdm(
        total=len(y_param_grid),
        desc="Сетка по параметрам: P(неподвижная точка)",
        disable=not show_grid_progress,
    )
    with _suppress_loky_grid_worker_warning():
        for i_y, row_probs in _parallel_for_grid_jobs(n_jobs, parallel_extra=parallel_extra)(
            delayed(worker)(i) for i in range(len(y_param_grid))
        ):
            P[i_y] = row_probs
            pbar.update(1)
    pbar.close()

    return P


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
        parallel_extra: Optional[Dict[str, Any]] = None,
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

    ``parallel_extra``: доп. аргументы для ``joblib.Parallel`` (см. ``_parallel_for_grid_jobs``).
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
        with _suppress_loky_grid_worker_warning():
            for i_y, row in _parallel_for_grid_jobs(n_jobs, parallel_extra=parallel_extra)(
                    delayed(worker)(i) for i in range(len(y_grid))):
                Z[i_y] = row
                pbar.update(1)

    return Z

