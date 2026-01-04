"""
Векторизованный один шаг RK4 для автономных систем с параметрами
"""
from .logger import get_logger

logger = get_logger(__name__)


def rk4_step_vectorized_params(f, state, parameters, h):
    k1 = f(state, parameters)
    k2 = f(state + h * k1 / 2, parameters)
    k3 = f(state + h * k2 / 2, parameters)
    k4 = f(state + h * k3, parameters)
    
    new_state = state + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)
    return new_state
