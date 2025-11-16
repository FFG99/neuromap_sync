import numpy as np
from utils.rk4 import rk4_step_vectorized_params

def van_der_pol_right_part(state, params):
    x, dx = state
    lambda_, w0_sq = params
    d2x = (lambda_ - x**2) * dx - w0_sq * x
    return np.array([dx, d2x])

def van_der_pol_rk4(state, params, dt):
    return rk4_step_vectorized_params(van_der_pol_right_part, state, params, dt)
