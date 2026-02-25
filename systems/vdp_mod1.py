import numpy as np
from utils.rk4 import rk4_step_vectorized_params
from utils.logger import get_logger

logger = get_logger(__name__)

def vdp_mod1_right_part(state, params):
    """
    Правая часть уравнений модели (3).
    state – [x1, x2]
    params – [λ, β]   (β ≥ 0)
    """
    x1, x2 = state
    lam, beta = params

    dx1 =  x2
    dx2 = lam * (1 - x1**2) * x2 - x1 + beta * x1**3

    return np.array([dx1, dx2])

def vdp_mod1_rk4(state, params, dt):
    """Обёртка RK4 для модели (3)."""
    return rk4_step_vectorized_params(vdp_mod1_right_part, state, params, dt)
