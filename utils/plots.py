import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from typing import Dict, Optional, Tuple

from .logger import get_logger

logger = get_logger(__name__)


def plot_trajectory(traj, 
    variables_names=None,
    title=None):
    
    if traj is None:
        logger.info("Траектория разбежалась")
        return

    if len(traj) == 0:
        raise ValueError("Траектория не может быть пустой")
    
    dimension = len(traj[0])
    match dimension:
        case 2:
            xs = [x for (x, y) in traj]
            ys = [y for (x, y) in traj]
            
            plt.scatter(xs, ys, s=0.1)
            if variables_names is None:
                plt.xlabel(r'$x$')
                plt.ylabel(r'$\dot{x}$')
            else:
                plt.xlabel(variables_names[0])
                plt.ylabel(variables_names[1])

            if title:
                plt.title(title)

            plt.tight_layout()
            plt.show()
        case 3:
            xs = [x[0] for x in traj]
            ys = [x[1] for x in traj]
            zs = [x[2] for x in traj]

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(xs, ys, zs, s=0.1)

            if variables_names is None:
                ax.set_xlabel(r'$x$')
                ax.set_ylabel(r'$y$')
                ax.set_zlabel(r'$z$')
            else:
                ax.set_xlabel(variables_names[0])
                ax.set_ylabel(variables_names[1])
                ax.set_zlabel(variables_names[2])

            if title:
                ax.set_title(title)

            plt.tight_layout()
            plt.show()
        case 4:
            xs  = [x[0] for x in traj]
            dxs = [x[1] for x in traj]
            ys  = [x[2] for x in traj]
            dys = [x[3] for x in traj]

            fig, axs = plt.subplots(1, 2, figsize=(16, 7))
            axs[0].scatter(xs, dxs, s=0.1)
            
            if variables_names is None:
                axs[0].set_xlabel(r'$x$')
                axs[0].set_ylabel(r'$\dot{x}$')
                axs[0].grid(True)
                axs[1].scatter(ys, dys, s=0.1)
                axs[1].set_xlabel(r'$y$')
                axs[1].set_ylabel(r'$\dot{y}$')
                axs[1].grid(True)
            else:
                axs[0].xlabel(variables_names[0])
                axs[0].ylabel(variables_names[1])
                axs[1].xlabel(variables_names[2])
                axs[1].ylabel(variables_names[3])

            if title:
                fig.suptitle(title)

            plt.tight_layout()
            plt.show()
        case _:
            raise ValueError(f"Траектория размерности {dimension} не поддерживается")


def plot_heatmap(x, y, Z, *, x_label='x', y_label='y', title=None, ax=None):
    own_fig = ax is None
    if own_fig:
        _, ax = plt.subplots(figsize=(10, 8))
    fig = ax.figure

    im = ax.imshow(Z, extent=[x[0], x[-1], y[0], y[-1]], origin='lower',
                   aspect='auto', interpolation='none', cmap='plasma')

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if title:
        ax.set_title(title)

    fig.colorbar(im, ax=ax)
    if own_fig:
        plt.tight_layout()


def visualize_dynamic_regime_grid(Z: np.ndarray,
                                  x_grid: np.ndarray,
                                  y_grid: np.ndarray,
                                  x_param_name: str = "x",
                                  y_param_name: str = "y",
                                  regime_mapping: Optional[Dict[str, str]] = None,
                                  regime_colors: Optional[Dict[str, str]] = None,
                                  cmap: str = "tab20",
                                  title: Optional[str] = None,
                                  figsize: Tuple[int, int] = (10, 8),
                                  save_path: Optional[str] = None,
                                  show: bool = True) -> None:
    """
    Визуализирует сетку динамических режимов с помощью imshow.
    """
    Z_str = Z.astype(str)
    
    unique_regimes = np.unique(Z_str[Z_str != "unknown"])
    
    if regime_mapping is None:
        regime_mapping = {r: r for r in unique_regimes}
        if "unknown" in Z_str:
            regime_mapping["unknown"] = "неизвестный"
    
    if regime_colors is None:
        cmap_obj = plt.get_cmap(cmap, len(regime_mapping))
        regime_colors = {regime: cmap_obj(i) for i, regime in enumerate(regime_mapping.keys())}
    
    unique_regimes = list(regime_mapping.keys())
    regime_to_int = {regime: idx for idx, regime in enumerate(unique_regimes)}
    
    Z_int = np.zeros(Z_str.shape, dtype=int)
    for regime, int_val in regime_to_int.items():
        Z_int[Z_str == regime] = int_val
    
    colors = [regime_colors[regime] for regime in unique_regimes]
    cmap_obj = ListedColormap(colors)
    
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(Z_int, aspect='auto', origin='lower',
                   extent=[x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()],
                   cmap=cmap_obj, vmin=-0.5, vmax=len(unique_regimes)-0.5)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, ticks=range(len(unique_regimes)))
    cbar.ax.set_yticklabels([regime_mapping[regime] for regime in unique_regimes])
    
    ax.set_xlabel(f'{x_param_name}')
    ax.set_ylabel(f'{y_param_name}')
    if title:
        ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Сохранено: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_compare_heatmaps(x, y, Z_ode, Z_nm, caption=None, *, x_label='x', y_label='y', cmap='plasma'):
    """
    Сравнение двух карт амплитуд рядом друг с другом с одним общим колорбаром
    
    Args:
        x: сетка по оси x
        y: сетка по оси y
        Z_ode: карта амплитуд для ODE (Original)
        Z_nm: карта амплитуд для Neuromap
        caption: опциональный общий заголовок
        x_label: подпись оси x
        y_label: подпись оси y
        cmap: цветовая карта
    """
    X, Y = np.meshgrid(x, y)
    
    vmin = min(np.nanmin(Z_ode), np.nanmin(Z_nm))
    vmax = max(np.nanmax(Z_ode), np.nanmax(Z_nm))
    
    fig, axs = plt.subplots(1, 2, figsize=(16, 7))
    
    cs1 = axs[0].contourf(X, Y, Z_ode, levels=50, cmap=cmap, vmin=vmin, vmax=vmax)
    axs[0].set_xlabel(x_label)
    axs[0].set_ylabel(y_label)
    axs[0].set_title('Original')
    
    cs2 = axs[1].contourf(X, Y, Z_nm, levels=50, cmap=cmap, vmin=vmin, vmax=vmax)
    axs[1].set_xlabel(x_label)
    axs[1].set_ylabel(y_label)
    axs[1].set_title('Neuromap')
    
    plt.tight_layout()
    cbar = fig.colorbar(cs2, ax=axs, location='right', pad=0.02, shrink=0.8)
    
    if caption:
        fig.suptitle(caption, fontsize=14)
    
    plt.show()


def plot_amplitude_basin(
    x,
    y,
    Z: np.ndarray,
    divergence_mask: Optional[np.ndarray] = None,
    *,
    x_label='x',
    y_label='y',
    title: Optional[str] = None,
    cmap: str = 'plasma',
    bad_color: str = 'black',
    diverge_label: str = 'Divergence',
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Тепловая карта амплитуды с отдельным цветом для расходимости.

    Рекомендуется передавать `Z`, где расходимость закодирована как `np.nan`,
    но также можно передать `divergence_mask`.
    """
    if divergence_mask is None:
        divergence_mask = ~np.isfinite(Z)

    Z_plot = np.ma.array(Z, mask=divergence_mask)

    finite_vals = Z[np.isfinite(Z)]
    if finite_vals.size == 0:
        # Случай "всё разошлось": чтобы не падать на min/max
        vmin_ = 0.0
        vmax_ = 1.0
    else:
        vmin_ = float(np.min(finite_vals)) if vmin is None else float(vmin)
        vmax_ = float(np.max(finite_vals)) if vmax is None else float(vmax)

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color=bad_color)

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    im = ax.imshow(
        Z_plot,
        extent=[x[0], x[-1], y[0], y[-1]],
        origin='lower',
        aspect='auto',
        interpolation='none',
        cmap=cmap_obj,
        vmin=vmin_,
        vmax=vmax_,
    )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)

    fig.colorbar(im, ax=ax, label='Amplitude', shrink=0.92, pad=0.02)

    ax.legend(
        handles=[mpatches.Patch(color=bad_color, label=diverge_label)],
        loc='upper right',
        frameon=True,
    )
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Сохранено: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_compare_amplitude_basins(
    x,
    y,
    Z_ode: np.ndarray,
    Z_nm: np.ndarray,
    caption: Optional[str] = None,
    *,
    x_label: str = 'x',
    y_label: str = 'y',
    cmap: str = 'plasma',
    bad_color: str = 'black',
    diverge_label: str = 'Divergence',
):
    """
    Сравнение двух тепловых карт амплитуд (ODE vs Neuromap) на общей шкале.
    Расходимость (np.nan) показывается отдельным цветом.
    """
    finite_ode = Z_ode[np.isfinite(Z_ode)]
    finite_nm = Z_nm[np.isfinite(Z_nm)]
    finite_all = np.concatenate([finite_ode, finite_nm], axis=0)

    if finite_all.size == 0:
        vmin_ = 0.0
        vmax_ = 1.0
    else:
        vmin_ = float(np.min(finite_all))
        vmax_ = float(np.max(finite_all))

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color=bad_color)

    fig, axs = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    ims = []

    for ax, Z, title in [(axs[0], Z_ode, 'Original'), (axs[1], Z_nm, 'Neuromap')]:
        divergence_mask = ~np.isfinite(Z)
        Z_plot = np.ma.array(Z, mask=divergence_mask)

        im = ax.imshow(
            Z_plot,
            extent=[x[0], x[-1], y[0], y[-1]],
            origin='lower',
            aspect='auto',
            interpolation='none',
            cmap=cmap_obj,
            vmin=vmin_,
            vmax=vmax_,
        )
        ims.append(im)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)

    # Один общий colorbar справа, без налезания на оси
    fig.colorbar(
        ims[-1],
        ax=axs,
        label='Amplitude',
        shrink=0.92,
        pad=0.03,
        location='right',
    )

    axs[0].legend(
        handles=[mpatches.Patch(color=bad_color, label=diverge_label)],
        loc='upper right',
        frameon=True,
    )

    if caption:
        fig.suptitle(caption, fontsize=14)

    plt.show()


def plot_compare_trajectories(*trajectories, 
                             labels=None, 
                             caption=None, 
                             variables_names=None,
                             colors=None,
                             figsize=None,
                             layout='overlay',
                             alpha=0.7,
                             point_size=0.5):
    """
    Универсальное сравнение любого количества траекторий любой размерности
    
    Args:
        *trajectories: произвольное количество траекторий (каждая может быть None для divergence)
        labels: список подписей для траекторий (по умолчанию 'Trajectory 1', 'Trajectory 2', ...)
        caption: общий заголовок графика
        variables_names: названия переменных для осей (по умолчанию зависит от размерности)
        colors: цвета для траекторий (по умолчанию автоматические)
        figsize: размер фигуры (по умолчанию зависит от размерности и количества траекторий)
        layout: 'overlay' - все траектории на одном графике, 'sidebyside' - отдельные графики рядом
        alpha: прозрачность точек (по умолчанию 0.7)
        point_size: размер точек (по умолчанию 0.5)
    """
    if not trajectories:
        raise ValueError("Необходимо передать хотя бы одну траекторию")
    
    # Определяем размерность по первой непустой траектории
    dimension = None
    for traj in trajectories:
        if traj is not None and len(traj) > 0:
            dimension = len(traj[0])
            break
    
    if dimension is None:
        logger.warning("Все траектории пустые или None")
        return
    
    # Генерируем подписи по умолчанию
    if labels is None:
        labels = [f'Trajectory {i+1}' for i in range(len(trajectories))]
    
    # Генерируем названия переменных по умолчанию
    if variables_names is None:
        if dimension == 2:
            variables_names = [r'$x$', r'$\dot{x}$']
        elif dimension == 3:
            variables_names = [r'$x$', r'$y$', r'$z$']
        elif dimension == 4:
            variables_names = [r'$x$', r'$\dot{x}$', r'$y$', r'$\dot{y}$']
        else:
            variables_names = [f'$x_{i+1}$' for i in range(dimension)]
    
    # Генерируем цвета по умолчанию
    if colors is None:
        import matplotlib.pyplot as plt
        cmap = plt.cm.tab10
        colors = [cmap(i % 10) for i in range(len(trajectories))]
    
    # Определяем размер фигуры по умолчанию
    if figsize is None:
        if layout == 'sidebyside':
            if dimension == 2:
                figsize = (5 * len(trajectories), 6)
            elif dimension == 3:
                figsize = (6 * len(trajectories), 8)
            elif dimension == 4:
                figsize = (16, 7)
            else:
                figsize = (5 * len(trajectories), 6)
        else:  # overlay
            if dimension == 2:
                figsize = (10, 8)
            elif dimension == 3:
                figsize = (12, 9)
            elif dimension == 4:
                figsize = (16, 7)
            else:
                figsize = (12, 8)
    
    # Обрабатываем разные размерности
    if dimension == 2:
        _plot_2d_trajectories(trajectories, labels, colors, variables_names, caption, figsize, layout, alpha, point_size)
    elif dimension == 3:
        _plot_3d_trajectories(trajectories, labels, colors, variables_names, caption, figsize, layout, alpha, point_size)
    elif dimension == 4:
        _plot_4d_trajectories(trajectories, labels, colors, variables_names, caption, figsize, alpha, point_size)
    else:
        raise ValueError(f"Траектории размерности {dimension} не поддерживаются")


def _plot_2d_trajectories(trajectories, labels, colors, variables_names, caption, figsize, layout, alpha, point_size):
    """Отображение 2D траекторий"""
    if layout == 'sidebyside':
        # Отдельные графики рядом
        valid_trajectories = [t for t in trajectories if t is not None and len(t) > 0]
        if len(valid_trajectories) == 0:
            logger.warning("Нет валидных траекторий для отображения")
            return
        
        fig, axs = plt.subplots(1, len(trajectories), figsize=figsize)
        if len(trajectories) == 1:
            axs = [axs]
        
        # Для вычисления общих пределов
        all_xs, all_ys = [], []
        for traj in valid_trajectories:
            xs = [point[0] for point in traj]
            ys = [point[1] for point in traj]
            all_xs.extend(xs)
            all_ys.extend(ys)
        
        x_min, x_max = (min(all_xs), max(all_xs)) if all_xs else (0, 1)
        y_min, y_max = (min(all_ys), max(all_ys)) if all_ys else (0, 1)
        
        # Добавляем небольшой отступ
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_margin = x_range * 0.05 if x_range > 0 else 0.1
        y_margin = y_range * 0.05 if y_range > 0 else 0.1
        
        for i, (traj, label, color) in enumerate(zip(trajectories, labels, colors)):
            ax = axs[i]
            
            if traj is None:
                # Элегантное отображение divergence
                ax.text(0.5, 0.5, 'Divergence', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=16, color='red', weight='bold',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.8))
                logger.info(f"{label}: divergence")
            elif len(traj) == 0:
                ax.text(0.5, 0.5, 'Empty', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=16, color='gray', style='italic')
                logger.warning(f"{label}: пустая траектория")
            else:
                xs = [point[0] for point in traj]
                ys = [point[1] for point in traj]
                ax.scatter(xs, ys, color=color, alpha=alpha, s=point_size, rasterized=True)
            
            ax.set_xlabel(variables_names[0], fontsize=12)
            ax.set_ylabel(variables_names[1], fontsize=12)
            ax.set_title(label, fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(x_min - x_margin, x_max + x_margin)
            ax.set_ylim(y_min - y_margin, y_max + y_margin)
        
    else:  # overlay
        fig, ax = plt.subplots(figsize=figsize)
        
        # Сначала отображаем валидные траектории
        for traj, label, color in zip(trajectories, labels, colors):
            if traj is not None and len(traj) > 0:
                xs = [point[0] for point in traj]
                ys = [point[1] for point in traj]
                ax.scatter(xs, ys, label=label, color=color, alpha=alpha, s=point_size, rasterized=True)
        
        # Затем добавляем divergence траектории в легенду
        for traj, label, color in zip(trajectories, labels, colors):
            if traj is None:
                ax.scatter([], [], label=f"{label} (divergence)", color=color, 
                          marker='x', s=50, alpha=0.8)
                logger.info(f"{label}: divergence")
        
        ax.set_xlabel(variables_names[0], fontsize=12)
        ax.set_ylabel(variables_names[1], fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(framealpha=0.9)
    
    if caption:
        if layout == 'sidebyside':
            fig.suptitle(caption, fontsize=16, weight='bold')
        else:
            ax.set_title(caption, fontsize=16, weight='bold')
    
    plt.tight_layout()
    plt.show()


def _plot_3d_trajectories(trajectories, labels, colors, variables_names, caption, figsize, layout, alpha, point_size):
    """Отображение 3D траекторий"""
    if layout == 'sidebyside':
        # Отдельные 3D графики рядом
        valid_trajectories = [t for t in trajectories if t is not None and len(t) > 0]
        if len(valid_trajectories) == 0:
            logger.warning("Нет валидных траекторий для отображения")
            return
        
        fig = plt.figure(figsize=figsize)
        
        # Для вычисления общих пределов
        all_xs, all_ys, all_zs = [], [], []
        for traj in valid_trajectories:
            xs = [point[0] for point in traj]
            ys = [point[1] for point in traj]
            zs = [point[2] for point in traj]
            all_xs.extend(xs)
            all_ys.extend(ys)
            all_zs.extend(zs)
        
        x_min, x_max = (min(all_xs), max(all_xs)) if all_xs else (0, 1)
        y_min, y_max = (min(all_ys), max(all_ys)) if all_ys else (0, 1)
        z_min, z_max = (min(all_zs), max(all_zs)) if all_zs else (0, 1)
        
        # Добавляем небольшой отступ
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = z_max - z_min
        x_margin = x_range * 0.05 if x_range > 0 else 0.1
        y_margin = y_range * 0.05 if y_range > 0 else 0.1
        z_margin = z_range * 0.05 if z_range > 0 else 0.1
        
        for i, (traj, label, color) in enumerate(zip(trajectories, labels, colors)):
            ax = fig.add_subplot(1, len(trajectories), i+1, projection='3d')
            
            if traj is None:
                # Элегантное отображение divergence в 3D
                ax.text(0.5, 0.5, 0.5, 'Divergence', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=14, color='red', weight='bold')
                logger.info(f"{label}: divergence")
            elif len(traj) == 0:
                ax.text(0.5, 0.5, 0.5, 'Empty', 
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=14, color='gray', style='italic')
                logger.warning(f"{label}: пустая траектория")
            else:
                xs = [point[0] for point in traj]
                ys = [point[1] for point in traj]
                zs = [point[2] for point in traj]
                ax.scatter(xs, ys, zs, color=color, alpha=alpha, s=point_size, rasterized=True)
            
            ax.set_xlabel(variables_names[0], fontsize=10)
            ax.set_ylabel(variables_names[1], fontsize=10)
            ax.set_zlabel(variables_names[2], fontsize=10)
            ax.set_title(label, fontsize=12, weight='bold')
            ax.set_xlim(x_min - x_margin, x_max + x_margin)
            ax.set_ylim(y_min - y_margin, y_max + y_margin)
            ax.set_zlim(z_min - z_margin, z_max + z_margin)
        
    else:  # overlay
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Сначала отображаем валидные траектории
        for traj, label, color in zip(trajectories, labels, colors):
            if traj is not None and len(traj) > 0:
                xs = [point[0] for point in traj]
                ys = [point[1] for point in traj]
                zs = [point[2] for point in traj]
                ax.scatter(xs, ys, zs, label=label, color=color, alpha=alpha, s=point_size, rasterized=True)
        
        # Затем добавляем divergence траектории в легенду
        for traj, label, color in zip(trajectories, labels, colors):
            if traj is None:
                ax.scatter([], [], [], label=f"{label} (divergence)", color=color, 
                          marker='x', s=50, alpha=0.8)
                logger.info(f"{label}: divergence")
        
        ax.set_xlabel(variables_names[0], fontsize=12)
        ax.set_ylabel(variables_names[1], fontsize=12)
        ax.set_zlabel(variables_names[2], fontsize=12)
        ax.legend(framealpha=0.9)
    
    if caption:
        if layout == 'sidebyside':
            fig.suptitle(caption, fontsize=16, weight='bold')
        else:
            ax.set_title(caption, fontsize=16, weight='bold')
    
    plt.tight_layout()
    plt.show()


def _plot_4d_trajectories(trajectories, labels, colors, variables_names, caption, figsize, alpha, point_size):
    """Отображение 4D траекторий как два 2D графика"""
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    
    # Для вычисления общих пределов
    all_x1, all_x2, all_y1, all_y2 = [], [], [], []
    
    for traj in trajectories:
        if traj is not None and len(traj) > 0:
            x1s = [point[0] for point in traj]
            x2s = [point[1] for point in traj]
            y1s = [point[2] for point in traj]
            y2s = [point[3] for point in traj]
            
            all_x1.extend(x1s)
            all_x2.extend(x2s)
            all_y1.extend(y1s)
            all_y2.extend(y2s)
    
    # Добавляем небольшие отступы
    if all_x1 and all_x2:
        x1_min, x1_max = min(all_x1), max(all_x1)
        x2_min, x2_max = min(all_x2), max(all_x2)
        x1_range = x1_max - x1_min
        x2_range = x2_max - x2_min
        x1_margin = x1_range * 0.05 if x1_range > 0 else 0.1
        x2_margin = x2_range * 0.05 if x2_range > 0 else 0.1
        axs[0].set_xlim(x1_min - x1_margin, x1_max + x1_margin)
        axs[0].set_ylim(x2_min - x2_margin, x2_max + x2_margin)
    
    if all_y1 and all_y2:
        y1_min, y1_max = min(all_y1), max(all_y1)
        y2_min, y2_max = min(all_y2), max(all_y2)
        y1_range = y1_max - y1_min
        y2_range = y2_max - y2_min
        y1_margin = y1_range * 0.05 if y1_range > 0 else 0.1
        y2_margin = y2_range * 0.05 if y2_range > 0 else 0.1
        axs[1].set_xlim(y1_min - y1_margin, y1_max + y1_margin)
        axs[1].set_ylim(y2_min - y2_margin, y2_max + y2_margin)
    
    for traj, label, color in zip(trajectories, labels, colors):
        if traj is None:
            # Элегантное отображение divergence
            axs[0].scatter([], [], label=f"{label} (divergence)", color=color, 
                          marker='x', s=50, alpha=0.8)
            axs[1].scatter([], [], color=color, marker='x', s=50, alpha=0.8)
            logger.info(f"{label}: divergence")
        elif len(traj) == 0:
            logger.warning(f"{label}: пустая траектория")
            continue
        else:
            x1s = [point[0] for point in traj]
            x2s = [point[1] for point in traj]
            y1s = [point[2] for point in traj]
            y2s = [point[3] for point in traj]
            
            axs[0].scatter(x1s, x2s, label=label, color=color, alpha=alpha, s=point_size, rasterized=True)
            axs[1].scatter(y1s, y2s, color=color, alpha=alpha, s=point_size, rasterized=True)
    
    axs[0].set_xlabel(variables_names[0], fontsize=12)
    axs[0].set_ylabel(variables_names[1], fontsize=12)
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(framealpha=0.9)
    
    axs[1].set_xlabel(variables_names[2], fontsize=12)
    axs[1].set_ylabel(variables_names[3], fontsize=12)
    axs[1].grid(True, alpha=0.3)
    
    if caption:
        fig.suptitle(caption, fontsize=16, weight='bold')
    
    plt.tight_layout()
    plt.show()
