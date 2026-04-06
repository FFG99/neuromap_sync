# -*- coding: utf-8 -*-
"""
Поиск неподвижных точек нейросетевого отображения u ↦ u + d(u, p) и скан по параметрам.

1) **Корни** — декартово произведение стартов, для каждого ``scipy.optimize.root`` на d(u,p)=0.

2) **Траектория** — то же множество стартов, затем итерация отображения u ← u + d(u,p);
   остановка, когда ‖d‖ не превышает заданный порог (аналог residual_tol).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from joblib import delayed
from tqdm import tqdm

from utils.trajectories import _parallel_for_grid_jobs, _suppress_loky_grid_worker_warning


def u0_tensor_from_axes(u_start_axes: Sequence[np.ndarray]) -> np.ndarray:
    """
    Декартово произведение одномерных сеток → матрица стартов (N, n_var), порядок ``ij``.
    """
    axes = [np.asarray(a, dtype=np.float64).ravel() for a in u_start_axes]
    if not axes:
        raise ValueError("u_start_axes не должен быть пустым")
    if any(len(x) < 1 for x in axes):
        raise ValueError("каждая ось должна содержать хотя бы один узел")
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([m.ravel() for m in mesh], axis=1)


def _u0_grid_tensor(
    u_bounds: Sequence[Tuple[float, float]],
    n_var: int,
    n_starts: int,
) -> np.ndarray:
    """
    Как раньше для совместимости ``collect_fixed_points_random_guesses``:
    равномерная сетка по числу стартов с усечением первых ``n_starts`` точек.
    """
    if n_starts < 1:
        raise ValueError("n_starts must be >= 1")
    k = int(np.ceil(n_starts ** (1.0 / n_var)))
    k = max(k, 2)
    lows = np.array([b[0] for b in u_bounds], dtype=np.float64)
    highs = np.array([b[1] for b in u_bounds], dtype=np.float64)
    axes = [np.linspace(lows[i], highs[i], k) for i in range(n_var)]
    return u0_tensor_from_axes(axes)[:n_starts]


def dedupe_fixed_points(points: Sequence[np.ndarray], unique_tol: float) -> List[np.ndarray]:
    """Оставить уникальные точки с порогом по евклидовой норме."""
    out: List[np.ndarray] = []
    for u in points:
        v = np.asarray(u, dtype=np.float64).ravel()
        if any(np.linalg.norm(v - w) < unique_tol for w in out):
            continue
        out.append(v.copy())
    return out


def _collect_fixed_points_from_u0(
    model,
    p: Sequence[float],
    u0: np.ndarray,
    *,
    root_tol: float = 1e-10,
    residual_tol: float = 1e-3,
    unique_tol: float = 1e-4,
    root_method: str = "hybr",
) -> List[np.ndarray]:
    p_arr = np.atleast_1d(np.asarray(p, dtype=np.float64))
    found: List[np.ndarray] = []
    for row in u0:
        u0_vec = np.asarray(row, dtype=np.float64).ravel()
        u_star, _ok, _ = model.find_fixed_point(
            p_arr, u0_vec, tol=root_tol, method=root_method
        )
        d, _ = model.compute_d_and_jacobian(u_star, p_arr)
        resnorm = float(np.linalg.norm(np.asarray(d, dtype=np.float64)))
        if resnorm > residual_tol:
            continue
        found.append(np.asarray(u_star, dtype=np.float64).ravel())
    return dedupe_fixed_points(found, unique_tol)


def trajectory_iter_to_fixed_point(
    model,
    p: Sequence[float],
    u0: np.ndarray,
    *,
    max_iter: int = 50_000,
    d_tol: float = 1e-3,
    divergence_threshold: float = 1e5,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Итерация дискретного отображения u ← u + d(u, p) (как в ``simulate``).

    Считаем, что достигли (приближённой) неподвижной точки, если ‖d(u,p)‖ ≤ ``d_tol``.
    При разходимости (норма состояния > ``divergence_threshold``) или исчерпании
    ``max_iter`` возвращается (None, meta).
    """
    p_arr = np.atleast_1d(np.asarray(p, dtype=np.float64))
    u = np.asarray(u0, dtype=np.float64).ravel().reshape(1, -1)
    p_2d = p_arr.reshape(1, -1)
    meta: Dict[str, Any] = {
        "n_iter": 0,
        "converged": False,
        "final_d_norm": None,
    }
    for k in range(max_iter):
        X = np.concatenate([u, p_2d], axis=1)
        d = model.predict(X)
        dn = float(np.linalg.norm(np.asarray(d, dtype=np.float64)))
        meta["final_d_norm"] = dn
        meta["n_iter"] = k + 1
        if dn <= d_tol:
            meta["converged"] = True
            u_star = np.asarray(u.ravel(), dtype=np.float64).copy()
            d_chk, _ = model.compute_d_and_jacobian(u_star, p_arr)
            meta["residual_norm_after_check"] = float(
                np.linalg.norm(np.asarray(d_chk, dtype=np.float64))
            )
            return u_star, meta
        u = u + d
        if float(np.linalg.norm(u)) > divergence_threshold:
            return None, meta
    return None, meta


def _collect_fixed_points_from_u0_trajectory(
    model,
    p: Sequence[float],
    u0: np.ndarray,
    *,
    max_iter: int = 50_000,
    d_tol: float = 1e-3,
    unique_tol: float = 1e-4,
    divergence_threshold: float = 1e5,
    verify_residual_tol: Optional[float] = None,
) -> List[np.ndarray]:
    """
    Для каждого старта — траекторная итерация; успешные точки проходят опциональную
    проверку ‖d‖ по ``compute_d_and_jacobian`` (если ``verify_residual_tol`` задан).
    """
    p_arr = np.atleast_1d(np.asarray(p, dtype=np.float64))
    found: List[np.ndarray] = []
    for row in u0:
        u_star, meta = trajectory_iter_to_fixed_point(
            model,
            p_arr,
            row,
            max_iter=max_iter,
            d_tol=d_tol,
            divergence_threshold=divergence_threshold,
        )
        if u_star is None or not meta.get("converged"):
            continue
        if verify_residual_tol is not None:
            d, _ = model.compute_d_and_jacobian(u_star, p_arr)
            if float(np.linalg.norm(np.asarray(d, dtype=np.float64))) > verify_residual_tol:
                continue
        found.append(np.asarray(u_star, dtype=np.float64).ravel())
    return dedupe_fixed_points(found, unique_tol)


def collect_fixed_points_grid_starts(
    model,
    p: Sequence[float],
    u_start_axes: Sequence[np.ndarray],
    *,
    root_tol: float = 1e-10,
    residual_tol: float = 1e-3,
    unique_tol: float = 1e-4,
    root_method: str = "hybr",
) -> List[np.ndarray]:
    """
    Старты — декартово произведение одномерных сеток ``u_start_axes`` (длина = ``n_var`` модели).
    """
    n_var = int(model.n_var)
    if len(u_start_axes) != n_var:
        raise ValueError(
            f"Ожидается {n_var} осей в u_start_axes, передано {len(u_start_axes)}"
        )
    u0 = u0_tensor_from_axes(u_start_axes)
    return _collect_fixed_points_from_u0(
        model,
        p,
        u0,
        root_tol=root_tol,
        residual_tol=residual_tol,
        unique_tol=unique_tol,
        root_method=root_method,
    )


def collect_fixed_points_trajectory_grid(
    model,
    p: Sequence[float],
    u_start_axes: Sequence[np.ndarray],
    *,
    max_iter: int = 50_000,
    d_tol: float = 1e-3,
    unique_tol: float = 1e-4,
    divergence_threshold: float = 1e5,
    verify_residual_tol: Optional[float] = None,
) -> List[np.ndarray]:
    """
    Те же старты, что у ``collect_fixed_points_grid_starts``, но неподвижная точка
    ищется итерацией u ← u + d до ‖d‖ ≤ ``d_tol``.
    """
    n_var = int(model.n_var)
    if len(u_start_axes) != n_var:
        raise ValueError(
            f"Ожидается {n_var} осей в u_start_axes, передано {len(u_start_axes)}"
        )
    u0 = u0_tensor_from_axes(u_start_axes)
    return _collect_fixed_points_from_u0_trajectory(
        model,
        p,
        u0,
        max_iter=max_iter,
        d_tol=d_tol,
        unique_tol=unique_tol,
        divergence_threshold=divergence_threshold,
        verify_residual_tol=verify_residual_tol,
    )


def collect_fixed_points_random_guesses(
    model,
    p: Sequence[float],
    u_bounds: Sequence[Tuple[float, float]],
    n_repeats: int = 100,
    *,
    rng: Optional[np.random.Generator] = None,
    root_tol: float = 1e-10,
    residual_tol: float = 1e-3,
    unique_tol: float = 1e-4,
    root_method: str = "hybr",
) -> List[np.ndarray]:
    """
    Устаревшее имя: сетка из ``u_bounds`` по прежнему правилу (см. ``_u0_grid_tensor``).
    Аргумент ``rng`` игнорируется.
    """
    n_var = int(model.n_var)
    u0 = _u0_grid_tensor(u_bounds, n_var, n_repeats)
    return _collect_fixed_points_from_u0(
        model,
        p,
        u0,
        root_tol=root_tol,
        residual_tol=residual_tol,
        unique_tol=unique_tol,
        root_method=root_method,
    )


def _nearest_metrics_from_points(
    model,
    p: Sequence[float],
    pts: List[np.ndarray],
    *,
    log_floor: float = 1e-30,
) -> Tuple[float, float, Dict[str, Any]]:
    meta: Dict[str, Any] = {
        "n_unique_fixed": len(pts),
        "fixed_points": pts,
        "u_star": None,
        "multipliers": None,
    }
    if not pts:
        return float("nan"), float("nan"), meta
    norms = [np.linalg.norm(u) for u in pts]
    i_min = int(np.argmin(norms))
    u_star = pts[i_min]
    meta["u_star"] = u_star
    mult = np.asarray(model.compute_fixed_point_multipliers(u_star, p), dtype=np.complex128)
    meta["multipliers"] = mult
    rho = float(np.max(np.abs(mult)))
    log_rho = float(np.log(max(rho, log_floor)))
    dist = float(np.linalg.norm(u_star))
    return dist, log_rho, meta


def nearest_origin_fixed_point_metrics(
    model,
    p: Sequence[float],
    u_start_axes: Sequence[np.ndarray],
    *,
    rng: Optional[np.random.Generator] = None,
    log_floor: float = 1e-30,
    **collect_kw: Any,
) -> Tuple[float, float, Dict[str, Any]]:
    """
    Среди найденных неподвижных точек (метод корней) — та, у которой минимальна ||u||.
    """
    collect_kw.pop("seed", None)
    pts = collect_fixed_points_grid_starts(model, p, u_start_axes, **collect_kw)
    return _nearest_metrics_from_points(model, p, pts, log_floor=log_floor)


def nearest_origin_fixed_point_metrics_trajectory(
    model,
    p: Sequence[float],
    u_start_axes: Sequence[np.ndarray],
    *,
    rng: Optional[np.random.Generator] = None,
    log_floor: float = 1e-30,
    **traj_kw: Any,
) -> Tuple[float, float, Dict[str, Any]]:
    """
    То же, что ``nearest_origin_fixed_point_metrics``, но точки собираются
    итерацией траектории до ‖d‖ ≤ ``d_tol`` (см. ``collect_fixed_points_trajectory_grid``).
    """
    traj_kw.pop("seed", None)
    pts = collect_fixed_points_trajectory_grid(model, p, u_start_axes, **traj_kw)
    return _nearest_metrics_from_points(model, p, pts, log_floor=log_floor)


def _grid_scan_neuromap_nearest_fixed_point_core(
    model,
    params: List[Union[np.ndarray, float, int]],
    u_start_axes: Sequence[np.ndarray],
    *,
    n_jobs: int = -1,
    x_param_index: Optional[int] = None,
    y_param_index: Optional[int] = None,
    show_grid_progress: bool = True,
    parallel_extra: Optional[Dict[str, Any]] = None,
    tqdm_desc: str = "Скан неподвижных точек (nearest ||u||)",
    metrics_for_p: Callable[[Any, List[float]], Tuple[float, float, Dict[str, Any]]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grid_indices = [i for i, p in enumerate(params) if isinstance(p, np.ndarray)]
    if len(grid_indices) != 2:
        raise ValueError(
            f"Нужны ровно два параметра-сетки (np.ndarray), найдено {len(grid_indices)}"
        )

    if x_param_index is None:
        x_param_idx = grid_indices[0]
    else:
        if x_param_index not in grid_indices:
            raise ValueError("x_param_index не указывает на параметр-сетку")
        x_param_idx = x_param_index

    if y_param_index is None:
        try:
            y_param_idx = next(idx for idx in grid_indices if idx != x_param_idx)
        except StopIteration:
            raise ValueError("Не найден второй параметр-сетки")
    else:
        if y_param_index not in grid_indices:
            raise ValueError("y_param_index не указывает на параметр-сетку")
        if y_param_index == x_param_idx:
            raise ValueError("x_param_index и y_param_index должны различаться")
        y_param_idx = y_param_index

    x_grid = np.asarray(params[x_param_idx], dtype=float)
    y_grid = np.asarray(params[y_param_idx], dtype=float)
    if x_grid.ndim != 1 or y_grid.ndim != 1:
        raise ValueError("Сетки параметров должны быть одномерными")

    def build_params_local(x_val: float, y_val: float) -> List[float]:
        row: List[float] = []
        for i, p in enumerate(params):
            if i == x_param_idx:
                row.append(float(x_val))
            elif i == y_param_idx:
                row.append(float(y_val))
            else:
                row.append(float(p))
        return row

    dist_Z = np.empty((len(y_grid), len(x_grid)), dtype=float)
    log_Z = np.empty_like(dist_Z)

    def worker(i_y: int) -> Tuple[int, np.ndarray, np.ndarray]:
        py = float(y_grid[i_y])
        row_d = np.empty(len(x_grid), dtype=float)
        row_l = np.empty(len(x_grid), dtype=float)
        for j, px in enumerate(x_grid):
            p_local = build_params_local(float(px), py)
            d, lg, _ = metrics_for_p(model, p_local)
            row_d[j] = d
            row_l[j] = lg
        return i_y, row_d, row_l

    pbar = tqdm(
        total=len(y_grid),
        desc=tqdm_desc,
        disable=not show_grid_progress,
    )
    with _suppress_loky_grid_worker_warning():
        for i_y, row_d, row_l in _parallel_for_grid_jobs(n_jobs, parallel_extra=parallel_extra)(
            delayed(worker)(i) for i in range(len(y_grid))
        ):
            dist_Z[i_y] = row_d
            log_Z[i_y] = row_l
            pbar.update(1)
    pbar.close()

    return dist_Z, log_Z, x_grid, y_grid


def grid_scan_neuromap_nearest_fixed_point(
    model,
    params: List[Union[np.ndarray, float, int]],
    u_start_axes: Sequence[np.ndarray],
    *,
    n_jobs: int = -1,
    x_param_index: Optional[int] = None,
    y_param_index: Optional[int] = None,
    show_grid_progress: bool = True,
    parallel_extra: Optional[Dict[str, Any]] = None,
    **collect_kw: Any,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    2D-скан по двум параметрам (остальные — скаляры в ``params``), поиск НТ методом корней.
    """
    collect_kw.pop("seed", None)

    def metrics_for_p(m: Any, p_local: List[float]) -> Tuple[float, float, Dict[str, Any]]:
        return nearest_origin_fixed_point_metrics(
            m, p_local, u_start_axes, **collect_kw
        )

    return _grid_scan_neuromap_nearest_fixed_point_core(
        model,
        params,
        u_start_axes,
        n_jobs=n_jobs,
        x_param_index=x_param_index,
        y_param_index=y_param_index,
        show_grid_progress=show_grid_progress,
        parallel_extra=parallel_extra,
        tqdm_desc="Скан неподвижных точек (nearest ||u||), корни",
        metrics_for_p=metrics_for_p,
    )


def grid_scan_neuromap_nearest_fixed_point_trajectory(
    model,
    params: List[Union[np.ndarray, float, int]],
    u_start_axes: Sequence[np.ndarray],
    *,
    n_jobs: int = -1,
    x_param_index: Optional[int] = None,
    y_param_index: Optional[int] = None,
    show_grid_progress: bool = True,
    parallel_extra: Optional[Dict[str, Any]] = None,
    **traj_kw: Any,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    То же поле, что ``grid_scan_neuromap_nearest_fixed_point``, но сбор НТ
    итерацией u ← u + d до ‖d‖ ≤ ``d_tol`` (аргументы как у ``collect_fixed_points_trajectory_grid``).
    """
    traj_kw.pop("seed", None)

    def metrics_for_p(m: Any, p_local: List[float]) -> Tuple[float, float, Dict[str, Any]]:
        return nearest_origin_fixed_point_metrics_trajectory(
            m, p_local, u_start_axes, **traj_kw
        )

    return _grid_scan_neuromap_nearest_fixed_point_core(
        model,
        params,
        u_start_axes,
        n_jobs=n_jobs,
        x_param_index=x_param_index,
        y_param_index=y_param_index,
        show_grid_progress=show_grid_progress,
        parallel_extra=parallel_extra,
        tqdm_desc="Скан неподвижных точек (nearest ||u||), траектория",
        metrics_for_p=metrics_for_p,
    )
