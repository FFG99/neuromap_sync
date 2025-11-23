import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import json
from .logger import get_logger

logger = get_logger(__name__)


def plot_trajectory(traj, 
    variables_names=None,
    title=None):
    
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

            plt.tight_layout()
            plt.show()
        case _:
            raise ValueError(f"Траектория размерности {dimension} не поддерживается")

def plot_heatmap(x, y, Z):
    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots(figsize=(10, 8))
    cs = ax.contourf(X, Y, Z, levels=50, cmap="plasma")

    cbar = fig.colorbar(cs, ax=ax)

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    plt.tight_layout()
    plt.show()


def plot_training_history(history, title=None, figsize=(10, 6), log_scale=False):
    """
    Построение графиков истории обучения
    
    Args:
        history: словарь с историей обучения (dict) или путь к JSON файлу (str/Path)
                 Должен содержать ключи: 'train_loss', 'val_loss', 'epoch'
        title: заголовок графика
        figsize: размер фигуры
        log_scale: использовать ли логарифмическую шкалу для оси Y
    """
    if isinstance(history, (str, Path)):
        history_path = Path(history)
        if not history_path.exists():
            raise FileNotFoundError(f"Файл истории не найден: {history_path}")
        with open(history_path, 'r', encoding='utf-8') as f:
            history = json.load(f)
    
    if not isinstance(history, dict):
        raise ValueError("history должен быть словарем или путем к JSON файлу")
    
    epochs = history.get('epoch', [])
    train_loss = history.get('train_loss', [])
    val_loss = history.get('val_loss', [])
    
    if not epochs:
        raise ValueError("История обучения пуста")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if train_loss:
        ax.plot(epochs, train_loss, label='Train Loss', linewidth=2, alpha=0.8)
    
    if val_loss:
        ax.plot(epochs, val_loss, label='Val Loss', linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Эпоха', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title(title or 'История обучения', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    if log_scale:
        ax.set_yscale('log')
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax
