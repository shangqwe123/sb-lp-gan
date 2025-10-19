import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures # Optional, for non-linear fitting
from sklearn.metrics import mean_squared_error
import torch.nn as nn
import torch.optim as optim

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
    plt.figure(figsize=(10, 6))
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

def plot_diffusion_trajectories_2d(sde, num_samples=10, num_time_steps=100):
    """
    Plots the forward and backward diffusion trajectories for a 2D Gaussian distribution
    using the SDE's q_sample and p_posterior methods.

    Args:
        sde: An SDE instance with q_sample and p_posterior methods.
        num_samples: Number of initial points to sample and track.
        num_time_steps: Number of time steps for the simulation.
    """
    # Set up time points
    t_span = torch.linspace(sde.t_min, sde.t_max, num_time_steps).to(sde.device)
    # Need time steps for p_posterior
    time_steps_forward = t_span
    time_steps_backward = torch.flip(t_span, dims=[0]) # Reverse time for backward

    # 1. Sample initial points from a 2D Gaussian (mean=0, variance=1)
    # Sample from a standard Gaussian for x0
    x0 = torch.randn(num_samples, 2).to(sde.device)
    # Target distribution center (e.g., another Gaussian center at 0) for x1 in Bridge SDE
    # Assuming target is N(0, I) for simple Gaussian visualization
    x1 = torch.zeros_like(x0).to(sde.device) # Assuming the target distribution is centered at 0

    # --- Simulate forward diffusion ---
    # q_sample samples x_t given x0, x1, and t.
    # To show trajectory, we need to sample at each time step.
    forward_trajectories = [x0.cpu().numpy()]
    current_x = x0 # Start from x0

    # Simulating forward path by applying noise sequentially might not directly use q_sample
    # which gives x_t from x0 and x1 directly.
    # A simple forward simulation adds noise based on SDE structure.
    # Let's use q_sample to get samples at different times assuming x0 is fixed.
    # This won't show a continuous path from a *single* realization of noise,
    # but rather the distribution state at different times.
    # To show a trajectory of a single particle, we'd need Euler-Maruyama with drift/diffusion.
    # Since sdes.py provides q_sample, let's plot snapshots at different times.

    forward_trajectories = []
    for t in time_steps_forward:
        # q_sample expects t as a batch, so unsqueeze
        t_batch = torch.ones(num_samples, device=sde.device) * t
        xt = sde.q_sample(t_batch, x0, x1) # Sample x_t from x0 and x1 at time t
        forward_trajectories.append(xt.cpu().numpy())

    forward_trajectories = np.stack(forward_trajectories, axis=0) # Shape: (num_time_steps, num_samples, 2)


    # --- Simulate backward process ---
    # p_posterior samples x_{t-dt} given x_t, x0, x1, t, s
    # We start from the end of the forward process (most noisy) and go backward in time.
    # We will need a 'model' or estimate for x0 and x1 at each step for p_posterior.
    # Since this is visualization without a trained model, let's assume we know the true x0 and x1.
    # In a real denoising process, x0 would be predicted by a model.
    # For visualization, we can *use* the original x0 and x1 to guide the backward process,
    # assuming the SDE's p_posterior is a reverse step towards x0 and x1.

    backward_trajectories = [forward_trajectories[-1]] # Start from the last state of the forward trajectory
    current_x_backward = torch.tensor(forward_trajectories[-1], device=sde.device)

    # Iterate backward in time
    for i in range(num_time_steps - 1):
        t_current = time_steps_backward[i]
        t_prev = time_steps_backward[i+1] # The time we are stepping to

        # p_posterior requires t (current time) and s (previous time) as batches
        t_current_batch = torch.ones(num_samples, device=sde.device) * t_current
        t_prev_batch = torch.ones(num_samples, device=sde.device) * t_prev

        # p_posterior needs x0 and x1. For visualization without a model, use the true x0 and x1.
        # In a real denoising sampler, x0 would be a prediction from a model.
        # We also set ot_ode=False to use the SDE based p_posterior.
        try:
            # p_posterior(t, s, x, x0, ot_ode=False, x1=None, delta_t=None)
            x_prev = sde.p_posterior(t_current_batch, t_prev_batch, current_x_backward, x0, ot_ode=False, x1=x1)
            backward_trajectories.append(x_prev.cpu().numpy())
            current_x_backward = x_prev
        except Exception as e:
            print(f"Error during backward simulation step {i} at time {t_current.item():.4f}: {e}")
            print("Stopping backward trajectory simulation.")
            backward_trajectories = backward_trajectories[:i+1] # Keep trajectories computed so far
            break


    backward_trajectories = np.stack(backward_trajectories, axis=0) # Shape: (num_time_steps, num_samples, 2)


    # 4. Plot trajectories
    plt.figure(figsize=(8, 8))

    # Plot forward trajectories (connecting the sampled points at different times)
    for i in range(num_samples):
        plt.plot(forward_trajectories[:, i, 0], forward_trajectories[:, i, 1], linestyle='-', marker='.', markersize=4, alpha=0.7, label=f'Forward Sample {i+1}')

    # Plot backward trajectories
    if backward_trajectories.shape[0] > 1: # Only plot if backward simulation was successful for more than one step
        for i in range(num_samples):
            plt.plot(backward_trajectories[:, i, 0], backward_trajectories[:, i, 1], linestyle='--', marker='x', markersize=4, alpha=0.7, label=f'Backward Sample {i+1}')
    else:
        print("Backward trajectories could not be plotted.")


    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title(f'Diffusion Trajectories in 2D ({sde.schedule_type} Schedule)')
    # plt.legend() # Too many samples for legend, uncomment if num_samples is small
    plt.grid(True)
    plt.axis('equal') # Equal scaling for x and y axes
    plt.scatter(x0.cpu().numpy()[:, 0], x0.cpu().numpy()[:, 1], color='red', s=50, label='Initial $x_0$', zorder=5)
    if backward_trajectories.shape[0] > 1:
        plt.scatter(backward_trajectories[-1, :, 0], backward_trajectories[-1, :, 1], color='green', s=50, label='Final $x_0$ (Backward)', zorder=5)
    plt.legend()

    # Plot target distribution points (x1)
    plt.scatter(x1.cpu().numpy()[:, 0], x1.cpu().numpy()[:, 1], color='blue', s=50, marker='*', label='Target $x_1$', zorder=5)

    # Save the plot
    plt.savefig(f'diffusion_trajectories_2d_{sde.schedule_type}.png')
    plt.savefig(f'diffusion_trajectories_2d_{sde.schedule_type}.pdf', bbox_inches='tight')
    plt.close()
    print(f"2D diffusion trajectories plot saved as diffusion_trajectories_2d_{sde.schedule_type}.png")

def plot_diffusion_trajectories_1d(sde, num_samples=50, num_time_steps=100, mode='lp'):
    """
    Plots the forward and backward diffusion trajectories for a 1D double Gaussian distribution
    with data on the x-axis and time on the y-axis, on separate subplots.

    Args:
        sde: An SDE instance with q_sample and p_posterior methods.
        num_samples: Number of initial points to sample and track.
        num_time_steps: Number of time steps for the simulation.
    """
    # Set up time points
    t_span = torch.linspace(sde.t_min, sde.t_max, num_time_steps).to(sde.device)
    time_steps_forward = t_span
    time_steps_backward = torch.flip(t_span, dims=[0]) # Reverse time for backward

    # 1. Sample initial points from a 1D double Gaussian distribution
    torch.manual_seed(42) # Set a fixed random seed for reproducibility
    mean = 2.0
    std = 0.7
    num_samples_per_mode = num_samples // 2
    x0_mode1 = torch.randn(num_samples_per_mode, 1) * std - mean
    x0_mode2 = torch.randn(num_samples - num_samples_per_mode, 1) * std + mean
    x0 = torch.cat([x0_mode1, x0_mode2], dim=0).to(sde.device) # Shape: (num_samples, 1)

    # Target distribution center (e.g., another Gaussian center at 0) for x1 in Bridge SDE
    x1 = torch.cat([x0_mode2, x0_mode1], dim=0).to(sde.device) # Shape: (num_samples, 1)
    # x1 = torch.zeros_like(x0).to(sde.device) # Shape: (num_samples, 1)

    # --- Simulate forward diffusion ---
    forward_trajectories = []
    train_x0 = []
    train_xt = []
    train_t = []
    for t in time_steps_forward:
        t_batch = torch.ones(num_samples, device=sde.device) * t
        xt = sde.q_sample(t_batch, x0, x1) # Sample x_t from x0 and x1 at time t
        forward_trajectories.append(xt.cpu().numpy())
        train_x0.append(x0)
        train_xt.append(xt)
        train_t.append(t_batch)
    
    train_x0 = torch.cat(train_x0,0)
    train_xt = torch.cat(train_xt,0)
    train_t = torch.cat(train_t,0)

    forward_trajectories = np.stack(forward_trajectories, axis=0) # Shape: (num_time_steps, num_samples, 1)

    # --- Train a simple PyTorch DNN model to predict x0 from xt and t ---
    print("Starting PyTorch DNN model training...")


    # Define a simple DNN model
    class SimpleDNN(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(SimpleDNN, self).__init__()
            self.fc1 = nn.Linear(input_dim, 64)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(64, 64)
            self.relu = nn.ReLU()
            self.fc3 = nn.Linear(64, output_dim)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            x = self.relu(x)
            x = self.fc3(x)
            return x

    # Prepare data for PyTorch (convert to tensors)
    X_train_tensor = torch.cat([train_xt, train_t.unsqueeze(-1)], dim=-1).to(sde.device) # Shape: (train_samples, 2)
    y_train_tensor = train_x0.to(sde.device) # Shape: (train_samples, 1)

    # Instantiate model, loss function, and optimizer
    input_dim = X_train_tensor.shape[-1]
    output_dim = y_train_tensor.shape[-1]
    model = SimpleDNN(input_dim, output_dim).to(sde.device)
    criterion = nn.MSELoss()

    if mode=='lp-train':
        prior = SimpleDNN(output_dim, output_dim).to(sde.device)
        optimizer = optim.Adam(list(model.parameters()) + list(prior.parameters()), lr=0.001)
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.001)

    if mode == 'lp-train':
        # Training loop
        num_epochs = 50 # Adjust epochs as needed
        for epoch in range(num_epochs):
            model.train()
            prior.train()
            optimizer.zero_grad()
            p1 = prior(x1)
            train_x0 = []
            train_xt = []
            train_t = []
            for t in time_steps_forward:
                t_batch = torch.ones(num_samples, device=sde.device) * t
                xt = sde.q_sample(t_batch, x0, p1) # Sample x_t from x0 and x1 at time t
                train_x0.append(x0)
                train_xt.append(xt)
                train_t.append(t_batch)
            
            train_x0 = torch.cat(train_x0,0)
            train_xt = torch.cat(train_xt,0)
            train_t = torch.cat(train_t,0)
            # Prepare data for PyTorch (convert to tensors)
            X_train_tensor = torch.cat([train_xt, train_t.unsqueeze(-1)], dim=-1).to(sde.device) # Shape: (train_samples, 2)
            y_train_tensor = train_x0.to(sde.device) # Shape: (train_samples, 1)

            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

            if (epoch+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')

        prior.eval()
        lp_x1 = prior(x1).detach()#.cpu().numpy()
        # --- Simulate forward diffusion ---
        forward_trajectories = []
        train_x0 = []
        train_xt = []
        train_t = []
        for t in time_steps_forward:
            t_batch = torch.ones(num_samples, device=sde.device) * t
            xt = sde.q_sample(t_batch, x0, lp_x1) # Sample x_t from x0 and x1 at time t
            forward_trajectories.append(xt.cpu().numpy())
            train_x0.append(x0)
            train_xt.append(xt)
            train_t.append(t_batch)
        
        train_x0 = torch.cat(train_x0,0)
        train_xt = torch.cat(train_xt,0)
        train_t = torch.cat(train_t,0)

        forward_trajectories = np.stack(forward_trajectories, axis=0) # Shape: (num_time_steps, num_samples, 1)

    else:
        # Training loop
        num_epochs = 5000 # Adjust epochs as needed
        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

            if (epoch+1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')
    print("PyTorch DNN model training finished.")
    print(f"PyTorch model training MSE: {loss.item():.6f}") # Print final training loss

    # --- Simulate backward process using the trained DNN model (randn=True) ---
    print("Simulating backward process with randn=True...")
    if mode=='lp-train':
        current_x_backward_randn_true = lp_x1 #torch.tensor(forward_trajectories[-1], dtype=torch.float32).to(sde.device) # Convert to tensor
        backward_trajectories_randn_true_list = [current_x_backward_randn_true] # Store as numpy for stacking
    else:
        current_x_backward_randn_true = x1 #torch.tensor(forward_trajectories[-1], dtype=torch.float32).to(sde.device) # Convert to tensor
        backward_trajectories_randn_true_list = [current_x_backward_randn_true.cpu().numpy()] # Store as numpy for stacking

    model.eval() # Set model to evaluation mode

    # Iterate backward in time
    for i in range(num_time_steps - 1):
        t_current = time_steps_backward[i] # Get scalar time for input feature
        t_prev = time_steps_backward[i+1] # The time we are stepping to

        # Prepare current data for model prediction
        current_t_batch_tensor = torch.full((num_samples, 1), t_current, device=sde.device)
        X_predict_tensor = torch.cat([current_x_backward_randn_true, current_t_batch_tensor], dim=-1) # Shape: (num_samples, 2)

        # Use the trained DNN model to predict x0
        with torch.no_grad():
            predicted_x0 = model(X_predict_tensor) # Shape: (num_samples, 1)

        try:
            # p_posterior(t, s, x, x0, ot_ode=False, x1=None, delta_t=None)
            x_prev = sde.p_posterior(t_prev, t_current, current_x_backward_randn_true, predicted_x0, ot_ode=False, x1=None, randn=True) # Pass predicted_x0, randn=True

            if x_prev.shape == current_x_backward_randn_true.shape:
                current_x_backward_randn_true = x_prev
                backward_trajectories_randn_true_list.append(current_x_backward_randn_true.cpu().numpy())
            else:
                print(f"Warning: Unexpected shape from p_posterior (randn=True) at step {i+1}/{num_time_steps-1} at time {t_current:.4f}. Expected {current_x_backward_randn_true.shape}, got {x_prev.shape}. Stopping backward trajectory simulation (randn=True).")
                break
        except Exception as e:
            print(f"Error during backward simulation (randn=True) step {i+1}/{num_time_steps-1} at time {t_current:.4f}: {e}")
            print("Stopping backward trajectory simulation (randn=True).")
            break

    backward_trajectories_randn_true = None
    if len(backward_trajectories_randn_true_list) > 1:
        try:
            backward_trajectories_randn_true = np.stack(backward_trajectories_randn_true_list, axis=0)
        except ValueError as e:
            print(f"Error stacking backward trajectories (randn=True) after simulation: {e}")
            print("Backward trajectories (randn=True) will not be plotted due to stacking issue.")

    # --- Simulate backward process using the trained DNN model (randn=False) ---
    print("Simulating backward process with randn=False...")
    current_x_backward_randn_false = torch.tensor(forward_trajectories[-1], dtype=torch.float32).to(sde.device)
    backward_trajectories_randn_false_list = [current_x_backward_randn_false.cpu().numpy()]

    model.eval() # Ensure model is in evaluation mode

    # Iterate backward in time
    for i in range(num_time_steps - 1):
        t_current = time_steps_backward[i]
        t_prev = time_steps_backward[i+1]

        # Prepare current data for model prediction
        current_t_batch_tensor = torch.full((num_samples, 1), t_current, device=sde.device)
        X_predict_tensor = torch.cat([current_x_backward_randn_false, current_t_batch_tensor], dim=-1)

        # Use the trained DNN model to predict x0
        with torch.no_grad():
            predicted_x0 = model(X_predict_tensor)

        try:
            # p_posterior(t, s, x, x0, ot_ode=False, x1=None, delta_t=None)
            x_prev = sde.p_posterior(t_prev, t_current, current_x_backward_randn_false, predicted_x0, ot_ode=False, x1=None, randn=False) # Pass predicted_x0, randn=False

            if x_prev.shape == current_x_backward_randn_false.shape:
                current_x_backward_randn_false = x_prev
                backward_trajectories_randn_false_list.append(current_x_backward_randn_false.cpu().numpy())
            else:
                print(f"Warning: Unexpected shape from p_posterior (randn=False) at step {i+1}/{num_time_steps-1} at time {t_current:.4f}. Expected {current_x_backward_randn_false.shape}, got {x_prev.shape}. Stopping backward trajectory simulation (randn=False).")
                break
        except Exception as e:
            print(f"Error during backward simulation (randn=False) step {i+1}/{num_time_steps-1} at time {t_current:.4f}: {e}")
            print("Stopping backward trajectory simulation (randn=False).")
            break

    backward_trajectories_randn_false = None
    if len(backward_trajectories_randn_false_list) > 1:
        try:
            backward_trajectories_randn_false = np.stack(backward_trajectories_randn_false_list, axis=0)
        except ValueError as e:
            print(f"Error stacking backward trajectories (randn=False) after simulation: {e}")
            print("Backward trajectories (randn=False) will not be plotted due to stacking issue.")

    # Determine overall min/max x-values for consistent scaling
    all_x_values = [forward_trajectories[:, :, 0]]
    if backward_trajectories_randn_true is not None:
        all_x_values.append(backward_trajectories_randn_true[:, :, 0])
    if backward_trajectories_randn_false is not None:
        all_x_values.append(backward_trajectories_randn_false[:, :, 0])

    all_x_values = np.concatenate(all_x_values)
    min_x = np.min(all_x_values)
    max_x = np.max(all_x_values)

    # 4. Plot trajectories on three subplots
    fig, axes = plt.subplots(1, 3, figsize=(21, 6)) # Increased figsize for three subplots

    num_samples_per_mode = num_samples // 2 # Get the split point

    # Plot forward trajectories (data on x-axis, time on y-axis)
    ax0 = axes[0]
    forward_trajectories_T = np.transpose(forward_trajectories, (1, 0, 2))
    for i in range(num_samples):
        # Forward trajectory color based on initial mode, linestyle: solid
        color = 'purple' if i < num_samples_per_mode else 'orange' # Use colors from the earlier attempt
        if mode=='lp-train':
            new_forward = np.concatenate([forward_trajectories_T[i, :, 0],x1[i].cpu().numpy()], 0)
            new_time = np.concatenate([ time_steps_forward.cpu().numpy(),[1.1]], 0)
            ax0.plot(new_forward, new_time, linestyle='-', alpha=0.5)#, color=color)
        else:
            ax0.plot(forward_trajectories_T[i, :, 0], time_steps_forward.cpu().numpy(), linestyle='-', alpha=0.5)#, color=color)


    ax0.set_xlabel('Data Value')
    ax0.set_ylabel('Time (t)')
    # ax0.set_title(f'Forward Diffusion Trajectories ({sde.schedule_type})')
    ax0.set_title(f'Forward Diffusion Trajectories')
    ax0.grid(True)
    # Scatter initial points with different markers based on mode
    ax0.scatter(x0[:num_samples_per_mode].cpu().numpy()[:, 0], torch.full((num_samples_per_mode,), sde.t_min).cpu().numpy(), color='red', s=50, marker='o', label='Initial $x_0$ (Mode 1)', zorder=5)
    ax0.scatter(x0[num_samples_per_mode:].cpu().numpy()[:, 0], torch.full((num_samples - num_samples_per_mode,), sde.t_min).cpu().numpy(), color='darkred', s=50, marker='*', label='Initial $x_0$ (Mode 2)', zorder=5)

    # Plot target distribution points (x1) with different markers based on corresponding initial mode
    # Need to use the same split logic as x0
    if mode=="lp-train":
        ax0.scatter(lp_x1[:num_samples_per_mode].cpu().numpy()[:, 0], torch.full((num_samples_per_mode,), sde.t_max).cpu().numpy(), color='green', s=50, marker='o', label='Given $x_1$ (corr. Mode 1 $x_0$)', zorder=5)
        ax0.scatter(lp_x1[num_samples_per_mode:].cpu().numpy()[:, 0], torch.full((num_samples - num_samples_per_mode,), sde.t_max).cpu().numpy(), color='darkgreen', s=50, marker='*', label='Given $x_1$ (corr. Mode 2 $x_0$)', zorder=5)
    else:
        ax0.scatter(x1[:num_samples_per_mode].cpu().numpy()[:, 0], torch.full((num_samples_per_mode,), sde.t_max).cpu().numpy(), color='green', s=50, marker='o', label='Given $x_1$ (corr. Mode 1 $x_0$)', zorder=5)
        ax0.scatter(x1[num_samples_per_mode:].cpu().numpy()[:, 0], torch.full((num_samples - num_samples_per_mode,), sde.t_max).cpu().numpy(), color='darkgreen', s=50, marker='*', label='Given $x_1$ (corr. Mode 2 $x_0$)', zorder=5)
    ax0.legend(loc='center right') # Adjust legend location

    # Add x1 points at t=1.1 if mode is lp-train
    if mode == 'lp-train':
        plot_time_1_1 = 1.1
        if x1.shape[0] == num_samples:
            ax0.scatter(x1[:num_samples_per_mode].cpu().numpy()[:, 0], torch.full((num_samples_per_mode,), plot_time_1_1).cpu().numpy(), color='blue', s=50, marker='o', label='Given $x_1$ (t=1.1, M1)', zorder=5)
            ax0.scatter(x1[num_samples_per_mode:].cpu().numpy()[:, 0], torch.full((num_samples - num_samples_per_mode,), plot_time_1_1).cpu().numpy(), color='darkblue', s=50, marker='*', label='Given $x_1$ (t=1.1, M2)', zorder=5)
        else:
            ax0.scatter(x1.cpu().numpy()[:, 0], torch.full((x1.shape[0],), plot_time_1_1).cpu().numpy(), color='blue', s=50, marker='X', label='Given $x_1$ (t=1.1)', zorder=5)


    # Plot backward trajectories (randn=True) - SDE Inference
    ax1 = axes[1]
    if backward_trajectories_randn_true is not None and backward_trajectories_randn_true.shape[0] > 1:
        simulated_time_steps_backward_randn_true = time_steps_backward[:backward_trajectories_randn_true.shape[0]].cpu().numpy()
        backward_trajectories_randn_true_T = np.transpose(backward_trajectories_randn_true, (1, 0, 2))

        for i in range(num_samples):
            # Backward trajectory color based on initial mode, linestyle: dashed
            color = 'purple' if i < num_samples_per_mode else 'orange' # Use colors from the 
            if mode=='lp-train':
                new_forward = np.concatenate([x1[i].cpu().numpy(),backward_trajectories_randn_true_T[i, :, 0]], 0)
                new_time = np.concatenate([[1.1], simulated_time_steps_backward_randn_true], 0)
                ax1.plot(new_forward, new_time, linestyle='-', alpha=0.5)#, color=color)
            else:
                ax1.plot(backward_trajectories_randn_true_T[i, :, 0], simulated_time_steps_backward_randn_true, linestyle='--', alpha=0.5)#, color=color)


        final_backward_time_randn_true = time_steps_backward[backward_trajectories_randn_true.shape[0]-1].item()
        # Scatter final backward points with different markers based on initial mode
        if backward_trajectories_randn_true.shape[1] == num_samples: # Check if shapes align
            ax1.scatter(backward_trajectories_randn_true[-1, :num_samples_per_mode, 0], torch.full((num_samples_per_mode,), final_backward_time_randn_true).cpu().numpy(), color='red', s=50, marker='o', label='Final $x_0$ (Bwd SDE, M1)', zorder=5)
            ax1.scatter(backward_trajectories_randn_true[-1, num_samples_per_mode:, 0], torch.full((num_samples - num_samples_per_mode,), final_backward_time_randn_true).cpu().numpy(), color='darkred', s=50, marker='*', label='Final $x_0$ (Bwd SDE, M2)', zorder=5)
        else:
             # Fallback if shapes don't match
             ax1.scatter(backward_trajectories_randn_true[-1, :, 0], torch.full((backward_trajectories_randn_true.shape[1],), final_backward_time_randn_true).cpu().numpy(), color='red', s=50, label='Final $x_0$ (Bwd SDE)', zorder=5)

        # Plot target distribution points (x1) with different markers based on corresponding initial mode
        if x1.shape[0] == num_samples: # Check if shapes align
            if mode=='lp-train':
                ax1.scatter(lp_x1[:num_samples_per_mode].cpu().numpy()[:, 0], torch.full((num_samples_per_mode,), sde.t_max).cpu().numpy(), color='green', s=50, marker='o', label='Given $x_1$ (corr. M1 $x_0$)', zorder=5)
                ax1.scatter(lp_x1[num_samples_per_mode:].cpu().numpy()[:, 0], torch.full((num_samples - num_samples_per_mode,), sde.t_max).cpu().numpy(), color='darkgreen', s=50, marker='*', label='Given $x_1$ (corr. M2 $x_0$)', zorder=5)
            else:
                ax1.scatter(x1[:num_samples_per_mode].cpu().numpy()[:, 0], torch.full((num_samples_per_mode,), sde.t_max).cpu().numpy(), color='green', s=50, marker='o', label='Given $x_1$ (corr. M1 $x_0$)', zorder=5)
                ax1.scatter(x1[num_samples_per_mode:].cpu().numpy()[:, 0], torch.full((num_samples - num_samples_per_mode,), sde.t_max).cpu().numpy(), color='darkgreen', s=50, marker='*', label='Given $x_1$ (corr. M2 $x_0$)', zorder=5)
        else:
            # Fallback if shapes don't match
            ax1.scatter(x1.cpu().numpy()[:, 0], torch.full((x1.shape[0],), sde.t_max).cpu().numpy(), color='green', s=50, marker='X', label='Given $x_1$', zorder=5)
        ax1.legend(loc='center right') # Adjust legend location

        # Add x1 points at t=1.1 if mode is lp-train
        if mode == 'lp-train':
            plot_time_1_1 = 1.1
            if x1.shape[0] == num_samples:
                ax1.scatter(x1[:num_samples_per_mode].cpu().numpy()[:, 0], torch.full((num_samples_per_mode,), plot_time_1_1).cpu().numpy(), color='blue', s=50, marker='o', label='Given $x_1$ (t=1.1, M1)', zorder=5)
                ax1.scatter(x1[num_samples_per_mode:].cpu().numpy()[:, 0], torch.full((num_samples - num_samples_per_mode,), plot_time_1_1).cpu().numpy(), color='darkblue', s=50, marker='*', label='Given $x_1$ (t=1.1, M2)', zorder=5)
            else:
                ax1.scatter(x1.cpu().numpy()[:, 0], torch.full((x1.shape[0],), plot_time_1_1).cpu().numpy(), color='blue', s=50, marker='X', label='Given $x_1$ (t=1.1)', zorder=5)

    else:
        ax1.text(0.5, 0.5, "Backward trajectories (randn=True) could not be plotted.",
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax1.transAxes, color='red')
        print("Backward trajectories (randn=True) could not be plotted successfully.")

    ax1.set_xlabel('Data Value')
    ax1.set_ylabel('Time (t)')
    # ax1.set_title(f'Backward Diffusion Trajectories ({sde.schedule_type}) (DNN Prediction, SDE Inference)')
    ax1.set_title(f'Backward Diffusion Trajectories (SDE Inference)')
    ax1.grid(True)

    # Plot backward trajectories (randn=False) - ODE Inference
    ax2 = axes[2]
    if backward_trajectories_randn_false is not None and backward_trajectories_randn_false.shape[0] > 1:
        simulated_time_steps_backward_randn_false = time_steps_backward[:backward_trajectories_randn_false.shape[0]].cpu().numpy()
        backward_trajectories_randn_false_T = np.transpose(backward_trajectories_randn_false, (1, 0, 2))

        for i in range(num_samples):
            # Backward trajectory color based on initial mode, linestyle: dashed
            color = 'purple' if i < num_samples_per_mode else 'orange' # Use colors from the earlier attempt
            if mode=='lp-train':
                new_forward = np.concatenate([x1[i].cpu().numpy(),backward_trajectories_randn_false_T[i, :, 0]], 0)
                new_time = np.concatenate([[1.1], simulated_time_steps_backward_randn_false], 0)
                ax2.plot(new_forward, new_time, linestyle='-', alpha=0.5)#, color=color)
            else:
                ax2.plot(backward_trajectories_randn_false_T[i, :, 0], simulated_time_steps_backward_randn_false, linestyle='--', alpha=0.5)#, color=color)

        final_backward_time_randn_false = time_steps_backward[backward_trajectories_randn_false.shape[0]-1].item()
        # Scatter final backward points with different markers based on initial mode
        if backward_trajectories_randn_false.shape[1] == num_samples: # Check if shapes align
            ax2.scatter(backward_trajectories_randn_false[-1, :num_samples_per_mode, 0], torch.full((num_samples_per_mode,), final_backward_time_randn_false).cpu().numpy(), color='red', s=50, marker='o', label='Final $x_0$ (Bwd ODE, M1)', zorder=5)
            ax2.scatter(backward_trajectories_randn_false[-1, num_samples_per_mode:, 0], torch.full((num_samples - num_samples_per_mode,), final_backward_time_randn_false).cpu().numpy(), color='darkred', s=50, marker='*', label='Final $x_0$ (Bwd ODE, M2)', zorder=5)
        else:
            # Fallback if shapes don't match
            ax2.scatter(backward_trajectories_randn_false[-1, :, 0], torch.full((backward_trajectories_randn_false.shape[1],), final_backward_time_randn_false).cpu().numpy(), color='red', s=50, label='Final $x_0$ (Backward, randn=False)', zorder=5)

        # Plot target distribution points (x1) with different markers based on corresponding initial mode
        if x1.shape[0] == num_samples: # Check if shapes align
            if mode=='lp-train':
                ax2.scatter(lp_x1[:num_samples_per_mode].cpu().numpy()[:, 0], torch.full((num_samples_per_mode,), sde.t_max).cpu().numpy(), color='green', s=50, marker='o', label='Given $x_1$ (corr. M1 $x_0$)', zorder=5)
                ax2.scatter(lp_x1[num_samples_per_mode:].cpu().numpy()[:, 0], torch.full((num_samples - num_samples_per_mode,), sde.t_max).cpu().numpy(), color='darkgreen', s=50, marker='*', label='Given $x_1$ (corr. M2 $x_0$)', zorder=5)
            else:
                ax2.scatter(x1[:num_samples_per_mode].cpu().numpy()[:, 0], torch.full((num_samples_per_mode,), sde.t_max).cpu().numpy(), color='green', s=50, marker='o', label='Given $x_1$ (corr. M1 $x_0$)', zorder=5)
                ax2.scatter(x1[num_samples_per_mode:].cpu().numpy()[:, 0], torch.full((num_samples - num_samples_per_mode,), sde.t_max).cpu().numpy(), color='darkgreen', s=50, marker='*', label='Given $x_1$ (corr. M2 $x_0$)', zorder=5)
        else:
            # Fallback if shapes don't match
            ax2.scatter(x1.cpu().numpy()[:, 0], torch.full((x1.shape[0],), sde.t_max).cpu().numpy(), color='green', s=50, marker='X', label='Given $x_1$', zorder=5)
        ax2.legend(loc='center right') # Adjust legend location

        # Add x1 points at t=1.1 if mode is lp-train
        if mode == 'lp-train':
            plot_time_1_1 = 1.1
            if x1.shape[0] == num_samples:
                ax2.scatter(x1[:num_samples_per_mode].cpu().numpy()[:, 0], torch.full((num_samples_per_mode,), plot_time_1_1).cpu().numpy(), color='blue', s=50, marker='o', label='Given $x_1$ (t=1.1, M1)', zorder=5)
                ax2.scatter(x1[num_samples_per_mode:].cpu().numpy()[:, 0], torch.full((num_samples - num_samples_per_mode,), plot_time_1_1).cpu().numpy(), color='darkblue', s=50, marker='*', label='Given $x_1$ (t=1.1, M2)', zorder=5)
            else:
                ax2.scatter(x1.cpu().numpy()[:, 0], torch.full((x1.shape[0],), plot_time_1_1).cpu().numpy(), color='blue', s=50, marker='X', label='Given $x_1$ (t=1.1)', zorder=5)

    else:
        ax2.text(0.5, 0.5, "Backward trajectories (randn=False) could not be plotted.",
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax2.transAxes, color='red')
        print("Backward trajectories (randn=False) could not be plotted successfully.")

    ax2.set_xlabel('Data Value')
    ax2.set_ylabel('Time (t)')
    # ax2.set_title(f'Backward Diffusion Trajectories ({sde.schedule_type}) (DNN Prediction, ODE Inference)')
    ax2.set_title(f'Backward Diffusion Trajectories (ODE Inference)')
    ax2.grid(True)

    # Set same x-axis limits for all subplots
    ax0.set_xlim(min_x, max_x)
    ax1.set_xlim(min_x, max_x)
    ax2.set_xlim(min_x, max_x)

    # Adjust y-axis limits to include 1.1
    max_y = max(sde.t_max, 1.1) + 0.05 # Add a small buffer
    ax0.set_ylim(sde.t_min - 0.05, max_y)
    ax1.set_ylim(sde.t_min - 0.05, max_y)
    ax2.set_ylim(sde.t_min - 0.05, max_y)

    plt.tight_layout()

    # Save the plot
    plt.savefig(f'diffusion_trajectories_1d_{sde.schedule_type}_subplots_dnn_two_backwards_{mode}.png')
    plt.savefig(f'diffusion_trajectories_1d_{sde.schedule_type}_subplots_dnn_two_backwards_{mode}.pdf', bbox_inches='tight')
    plt.close()
    print(f"1D diffusion trajectories plot saved as diffusion_trajectories_1d_{sde.schedule_type}_subplots_dnn_two_backwards_{mode}.png")
