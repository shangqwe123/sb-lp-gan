import torch
import matplotlib.pyplot as plt
import numpy as np

# Set font sizes
plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12

def plot_diffusion_coefficients(sde, t_points=100):
    """
    Plot the evolution of w_x, w_y, and var over time for a DiffusionBridgeSDE.
    
    Args:
        sde: DiffusionBridgeSDE instance
        t_points: Number of time points to evaluate
    """
    # Generate time points from t_min to t_max
    t = torch.linspace(sde.t_min, sde.t_max+sde.t_min, t_points)
    
    # Calculate coefficients
    alpha_t = sde.alpha_t(t)
    sigma_t = sde.sigma_t(t)
    sigma_T = sde.sigma_t(torch.ones_like(t))
    alpha_T = sde.alpha_t(torch.ones_like(t))
    
    # Calculate w_x, w_y, and var
    w_x = alpha_t * (sigma_T**2 - sigma_t**2) / sigma_T**2
    w_y = (alpha_t / alpha_T) * (sigma_t**2 / sigma_T**2)
    var = (alpha_t**2 * (sigma_T**2 - sigma_t**2) * sigma_t**2) / sigma_T**2
    
    # Create the plot
    plt.figure(figsize=(10, 4.7))
    plt.plot(t.numpy(), w_x.numpy(), label='w_x', linewidth=2)
    plt.plot(t.numpy(), w_y.numpy(), label='w_y', linewidth=2)
    plt.plot(t.numpy(), var.numpy(), label='var', linewidth=2)

    print(w_x, w_y, w_x+w_y)
    
    plt.xlabel('Time (t)')
    plt.ylabel('Coefficient Value')
    plt.title(f'Evolution of Diffusion Coefficients ({sde.schedule_type} Schedule)')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plt.savefig(f'diffusion_coefficients_{sde.schedule_type}.png')
    plt.savefig(f'diffusion_coefficients_{sde.schedule_type}.pdf', bbox_inches='tight')

    plt.close()
