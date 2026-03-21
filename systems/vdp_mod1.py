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
    x1, x2 = state          # x1 = x,   x2 = x'
    lam, beta = params      # λ (lam) и β
    omega = 5.1             # ω

    dx1 = x2

    dx2 = (lam + x1**2 - beta * x1**4) * x2 - omega**2 * x1

    return np.array([dx1, dx2])

def vdp_mod1_rk4(state, params, dt):
    """Обёртка RK4 для модели (3)."""
    return rk4_step_vectorized_params(vdp_mod1_right_part, state, params, dt)
