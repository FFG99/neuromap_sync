import numpy as np
from utils.rk4 import rk4_step_vectorized_params
from utils.logger import get_logger

logger = get_logger(__name__)


def coupled_oscillators_right_part(state, params):
    """
    Правая часть для 4-мерной системы связанных осцилляторов Ван дер Поля.
    
    Args:
        state: [x1, dx1, x2, dx2] - состояние системы
        params: [λ1, ω₀²₁, λ2, ω₀²₂, ε] - параметры системы
    
    Returns:
        np.array: [dx1/dt, d²x1/dt², dx2/dt, d²x2/dt²]
    """
    x1, dx1, x2, dx2 = state
    lambda1, w0_sq1, lambda2, w0_sq2, epsilon = params
    
    # Первый осциллятор
    d2x1 = (lambda1 - x1**2) * dx1 - w0_sq1 * x1 - epsilon * (dx1 - dx2)
    
    # Второй осциллятор
    d2x2 = (lambda2 - x2**2) * dx2 - w0_sq2 * x2 - epsilon * (dx2 - dx1)
    
    return np.array([dx1, d2x1, dx2, d2x2])


def coupled_oscillators_rk4(state, params, dt):
    """
    Оператор эволюции для 4-мерной системы связанных осцилляторов.
    Использует метод Рунге-Кутты 4-го порядка.
    
    Args:
        state: [x1, dx1, x2, dx2] - состояние системы
        params: [λ1, ω₀²₁, λ2, ω₀²₂, ε] - параметры системы
        dt: шаг времени
    
    Returns:
        np.array: следующее состояние системы
    """
    return rk4_step_vectorized_params(coupled_oscillators_right_part, state, params, dt)

