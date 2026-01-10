import numpy as np
from utils.rk4 import rk4_step_vectorized_params
from utils.logger import get_logger

logger = get_logger(__name__)

def chua_right_part(state, params):
    x, y, z = state
    alpha, beta, gamma, m0, m1 = params
    f     = m1 * x + (m0-m1) * (abs(x+1)-abs(x-1)) / 2
    x_dot = alpha * (y - x) - alpha*f
    y_dot =  x - y + z
    z_dot =  - (beta * y + gamma * z)
    return np.array([x_dot, y_dot, z_dot])


def chua_rk4(state, params, dt):
    return rk4_step_vectorized_params(chua_right_part, state, params, dt)
