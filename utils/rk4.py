import numpy as np

def rk4_step_vectorized_params(f, t, state, parameters, h):
    """
    Векторизованный один шаг RK4 для систем с параметрами
    
    Parameters:
    f - функция: f(t, state, parameters) -> производные
    t - текущее время
    state - текущее состояние (может быть вектором или матрицей)
    parameters - параметры системы
    h - шаг интегрирования
    
    Returns:
    new_state - новое состояние после одного шага
    """
    k1 = f(t, state, parameters)
    k2 = f(t + h/2, state + h * k1 / 2, parameters)
    k3 = f(t + h/2, state + h * k2 / 2, parameters)
    k4 = f(t + h, state + h * k3, parameters)
    
    new_state = state + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)
    return new_state
