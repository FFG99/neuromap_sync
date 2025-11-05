import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from .trajectories import get_attractor_trajectory


def plot_bifurcation_diagram(evolution_operator, state, params, dt,
                           n_transient,
                           n_attractor,
                           changing_parameter_number,
                           changing_parameter_range,
                           secant_plane,
                           changing_parameter_num=50,
                           direction='increase',
                           project_on_basis=0,
                           changing_parameter_name=r'$\lambda$',
                           projection_variable_name=r'$x$',
                           point_size=0.5,
                           alpha=0.6):

    
    changing_parameter_min, changing_parameter_max = changing_parameter_range
    
    if direction == 'decrease':
        parameters = np.linspace(changing_parameter_max, changing_parameter_min, changing_parameter_num)
    else:
        parameters = np.linspace(changing_parameter_min, changing_parameter_max, changing_parameter_num)
    
    xs, ys = [], []
    local_state = state.copy()
    
    for parameter in tqdm(parameters, desc="Computing bifurcation diagram"):
        local_params = params.copy()
        local_params[changing_parameter_number] = parameter
        
        traj = get_attractor_trajectory(
            evolution_operator=evolution_operator,
            state=local_state, 
            params=local_params, 
            dt=dt,
            n_transient=n_transient, 
            n_attractor=n_attractor, 
            secant_plane=secant_plane
        )
        
        for point in traj:
            xs.append(parameter)
            ys.append(point[project_on_basis])
        
        local_state = traj[-1]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(xs, ys, s=point_size, c='darkblue', alpha=alpha, rasterized=True)
    
    ax.set_xlabel(changing_parameter_name, fontsize=14)
    ax.set_ylabel(projection_variable_name, fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Добавляем стрелку направления изменения параметра
    x_center = (changing_parameter_max + changing_parameter_min) / 2
    x_range = changing_parameter_max - changing_parameter_min
    y_min, y_max = ax.get_ylim()
    arrow_length = x_range * 0.05  # Длина стрелки как 5% от диапазона x    
    if direction == 'increase':
        arrow_dx = arrow_length
    else:
        arrow_dx = -arrow_length
    arrow_x = changing_parameter_min + x_range * 0.05
    arrow_y = y_min + (y_max - y_min) * 0.05
    if direction == 'decrease':
        arrow_x = changing_parameter_min + x_range * 0.05 + arrow_length
    
    ax.annotate('', xy=(arrow_x + arrow_dx, arrow_y), 
                xytext=(arrow_x, arrow_y),
                arrowprops=dict(arrowstyle='->', lw=2, color='darkblue'),
                annotation_clip=False)
    
    plt.tight_layout()
    plt.show()
