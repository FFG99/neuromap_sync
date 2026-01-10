import numpy as np
from matplotlib import pyplot as plt
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

def plot_compare(traj_ode, traj_nm, caption=None):
    """
    Сравнение двух 2D траекторий рядом друг с другом
    
    Args:
        traj_ode: траектория ODE (Original)
        traj_nm: траектория Neuromap
        caption: опциональный общий заголовок
    """
    if len(traj_ode) == 0 or len(traj_nm) == 0:
        raise ValueError("Траектории не могут быть пустыми")
    
    dim_ode = len(traj_ode[0])
    dim_nm = len(traj_nm[0])
    
    if dim_ode != 2 or dim_nm != 2:
        raise ValueError(f"plot_compare поддерживает только 2D траектории. Получены размерности: {dim_ode} и {dim_nm}")
    
    xs_ode = [x for (x, y) in traj_ode]
    ys_ode = [y for (x, y) in traj_ode]
    
    xs_nm = [x for (x, y) in traj_nm]
    ys_nm = [y for (x, y) in traj_nm]
    
    # Вычисляем общие пределы для обеих траекторий
    x_min = min(min(xs_ode), min(xs_nm))
    x_max = max(max(xs_ode), max(xs_nm))
    y_min = min(min(ys_ode), min(ys_nm))
    y_max = max(max(ys_ode), max(ys_nm))
    
    fig, axs = plt.subplots(1, 2, figsize=(16, 7))
    
    axs[0].scatter(xs_ode, ys_ode, s=0.1)
    axs[0].set_xlabel(r'$x$')
    axs[0].set_ylabel(r'$\dot{x}$')
    axs[0].set_title('Original')
    axs[0].grid(True)
    axs[0].set_xlim(x_min, x_max)
    axs[0].set_ylim(y_min, y_max)
    
    axs[1].scatter(xs_nm, ys_nm, s=0.1)
    axs[1].set_xlabel(r'$x$')
    axs[1].set_ylabel(r'$\dot{x}$')
    axs[1].set_title('Neuromap')
    axs[1].grid(True)
    axs[1].set_xlim(x_min, x_max)
    axs[1].set_ylim(y_min, y_max)
    
    if caption:
        fig.suptitle(caption, fontsize=14)
    
    plt.tight_layout()
    plt.show()


def plot_heatmap(x, y, Z, *, x_label='x', y_label='y'):
    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots(figsize=(10, 8))
    cs = ax.contourf(X, Y, Z, levels=50, cmap="plasma")

    cbar = fig.colorbar(cs, ax=ax)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    plt.tight_layout()
    plt.show()


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
