import numpy as np
from utils.rk4 import rk4_step_vectorized_params
from utils.logger import get_logger

logger = get_logger(__name__)

def rossler_right_part(state, params):
    x, y, z = state
    a, b, c = params
    x_dot = - y - z
    y_dot =  x + a*y
    z_dot =  b + z * (x - c)
    return np.array([x_dot, y_dot, z_dot])


def rossler_rk4(state, params, dt):
    return rk4_step_vectorized_params(rossler_right_part, state, params, dt)
