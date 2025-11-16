from matplotlib import pyplot as plt

def plot_trajectory(traj, 
    variables_names=None):
    if len(traj) == 0:
        raise ValueError
    
    dimension = len(traj[0])
    match dimension:
        case 2:
            xs = [x for (x, y) in traj]
            ys = [y for (x, y) in traj]
            
            plt.scatter(xs, ys, s=0.1)
            if variables_names is None:
                plt.xlabel(r'$x$')
                plt.ylabel(r'$\dot{x}$')
            else:
                plt.xlabel(variables_names[0])
                plt.ylabel(variables_names[1])

            plt.tight_layout()
            plt.show()
        case 4:
            xs  = [x[0] for x in traj]
            dxs = [x[1] for x in traj]
            ys  = [x[2] for x in traj]
            dys = [x[3] for x in traj]

            fig, axs = plt.subplots(1, 2, figsize=(16, 7))
            axs[0].scatter(xs, dxs, s=0.1)
            
            if variables_names is None:
                axs[0].set_xlabel(r'$x$')
                axs[0].set_ylabel(r'$\dot{x}$')
                axs[0].grid(True)
                axs[1].scatter(ys, dys, s=0.1)
                axs[1].set_xlabel(r'$y$')
                axs[1].set_ylabel(r'$\dot{y}$')
                axs[1].grid(True)
            else:
                axs[0].xlabel(variables_names[0])
                axs[0].ylabel(variables_names[1])
                axs[1].xlabel(variables_names[2])
                axs[1].ylabel(variables_names[3])

            plt.tight_layout()
            plt.show()
        case _:
            raise ValueError(f"Trajectory dimension={dimension} is not supported")
