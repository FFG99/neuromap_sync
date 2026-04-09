import numpy as np
from utils.rk4 import rk4_step_vectorized_params
from utils.logger import get_logger

logger = get_logger(__name__)

def vdp_mod2_right_part(state, params):
    """
    Правая часть уравнений модели (4).
    state – [x1, x2]
    params – [λ, μ]   (μ ≥ 0)
    """
    x1, x2 = state
    lam, mu = params
    omega = 5.1

    dx1 =  x2
    dx2 = (lam + mu * x1**2 - x1**4) * x2 - omega**2 * x1

    return np.array([dx1, dx2])

def vdp_mod2_rk4(state, params, dt):
    """Обёртка RK4 для модели (4)."""
    return rk4_step_vectorized_params(vdp_mod2_right_part, state, params, dt)
