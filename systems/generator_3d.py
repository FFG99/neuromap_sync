import numpy as np
from utils.rk4 import rk4_step_vectorized_params
from utils.logger import get_logger

logger = get_logger(__name__)

def generator_3d_right_part(state, params):
    x, y, z = state
    lambda_, beta, w0, k = params
    x_dot = y
    y_dot = (lambda_ + z + x**2 - beta * x**4) * y - w0**2 * x
    z_dot = -z - k * y**2
    return np.array([x_dot, y_dot, z_dot])


def generator_3d_rk4(state, params, dt):
    return rk4_step_vectorized_params(generator_3d_right_part, state, params, dt)