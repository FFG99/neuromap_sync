import numpy as np


def pass_transient_process(evolution_operator, state, params, dt, 
                           required_number_of_intersections, secant_plane,
                           max_steps=100_000_000) -> np.ndarray | None:
    
    number_of_intersections = 0
    step_count = 0
    current_state = np.array(state, dtype=np.float64)
    previous_state = None
    
    while number_of_intersections < required_number_of_intersections:
        step_count += 1
        if step_count > max_steps:
            return None
        
        previous_state = current_state
        current_state = evolution_operator(current_state, params, dt)
            
        if previous_state is not None:
            S_prev = secant_plane(previous_state)
            S_curr = secant_plane(current_state)
            
            if S_prev < 0 and S_curr >= 0:
                number_of_intersections += 1
                
    return current_state


def get_attractor_trajectory(evolution_operator, state, params, dt, 
                             n_transient, n_attractor, secant_plane,
                             accuracy=1e-4, max_steps=100_000_000,
                             fixed_point_threshold=1e-12):

    state = pass_transient_process(evolution_operator, state, params,
                                   dt, n_transient, secant_plane, max_steps)

    attractor_trajectory = []
    number_of_intersections = 0
    previous_state = None
    first_point = None

    while number_of_intersections < n_attractor:
        previous_state = state
        state = evolution_operator(state, params, dt)
        
        if np.linalg.norm(state - previous_state) < fixed_point_threshold:
            return [state]
        
        attractor_trajectory.append(state)
        S_prev = secant_plane(previous_state)
        S_curr = secant_plane(state)

        if S_prev < 0 and S_curr >= 0:
            dS = -S_curr
            sect_point = evolution_operator(state, params, dS)

            if first_point is None:
                first_point = sect_point
            else:
                if np.linalg.norm(sect_point - first_point) < accuracy:
                    return attractor_trajectory
            
            number_of_intersections += 1

    return attractor_trajectory

