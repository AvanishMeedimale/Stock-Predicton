# Program to simulate the SDE for specific parameters
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Numerical method to simulate Ornstein-Uhlenbeck process


def euler_maruyama(dt, n_steps, alpha, mu, sigma):
    dW = np.random.normal(0, np.sqrt(dt), n_steps)
    X = np.zeros(n_steps)
    X[0] = mu

    for i in range(1, n_steps):
        X[i] = X[i - 1] + alpha * (mu - X[i - 1]) * dt + sigma * dW[i]

    return X


data = pd.read_csv("F.csv")
stock_value = data['Close'].values
dt = 0.01
n_steps = 1000
alpha_mle = 0.2751004734327965
mu_mle = 28.456623583118425
sigma_mle = 2.2469816125876734
T = dt * n_steps

# Assuming your observed data has a time step of 1.0 (adjust if necessary)
dt_observed = 1.0
n_steps_observed = len(stock_value)
T_observed = dt_observed * n_steps_observed

# Set simulation parameters
dt_simulated = 1.0  # Use the same time step as the observed data
# Match the number of time steps with the observed data
n_steps_simulated = n_steps_observed
T_simulated = dt_simulated * n_steps_simulated

# Simulate multiple paths
n_simulations = 10
simulated_paths = []
for _ in range(n_simulations):
    X_simulated = euler_maruyama(
        dt_simulated, n_steps_simulated, alpha_mle, mu_mle, sigma_mle)
    simulated_paths.append(X_simulated)

# Create time array for plotting
time_steps_observed = np.arange(0, T_observed, dt_observed)
time_steps_simulated = np.arange(0, T_simulated, dt_simulated)

n_rows = 2  # Number of rows of subplots
n_cols = 5  # Number of columns of subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 6), sharex=True)


# Plot simulated paths in subplots
for i in range(n_simulations):
    row = i // n_cols
    col = i % n_cols
    axes[row, col].plot(time_steps_observed, stock_value,
                        label='Observed Data', color='blue')
    axes[row, col].plot(time_steps_simulated, simulated_paths[i],
                        label=f'Simulation {i+1}', alpha=0.7)
    axes[row, col].set_ylabel('Stock Price')
    axes[row, col].grid(True)

# Adjust layout and labels of subplots
for ax in axes.flat:
    ax.label_outer()
    ax.set_xlabel('Time')

plt.tight_layout()
plt.show()
