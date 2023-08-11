import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Reading the stock price data and extract daily stock prices
data = pd.read_csv("PXWEX.csv")
stock_value = data['Close'].values
dt = 1.0
n_steps = len(stock_value)
total_step_length = dt * n_steps

# Defining the log likelihood function to calculate probability of parameters fitting data


def log_likelihood(params, stock_price, dt):
    log_alpha, mu, sigma = params
    # Log alpha taken to avoid division by small alpha values
    alpha = np.exp(log_alpha)
    ll = 0.0
    for i in range(1, n_steps):
        conditional_mean = stock_price[i-1] * \
            np.exp(-alpha*dt) + mu*(1-np.exp(-alpha*dt))
        conditional_variance = (sigma**2 / (2 * alpha)) * \
            (1 - np.exp(-2 * alpha * dt))
        density = (1 / np.sqrt(2 * np.pi * conditional_variance)) * \
            np.exp(-((stock_price[i] - conditional_mean)
                   ** 2) / (2 * conditional_variance))
        ll += np.log(density)

    return -ll


# Maximizing the log likelihood function using L-BFGS-B optimization algorithm
initial_params = [1.0, np.mean(stock_value), np.std(stock_value)]
result = minimize(log_likelihood, initial_params,
                  args=(stock_value, dt), method='L-BFGS-B',
                  bounds=[(-5, None), (0, None), (0.05, None)])
log_alpha_mle, mu_mle, sigma_mle = result.x
alpha_mle = np.exp(log_alpha_mle)

# Estimating the most probable SDE parameters to fit the stock data
print("Estimated alpha:", alpha_mle)
print("Estimated mu:", mu_mle)
print("Estimated sigma:", sigma_mle)

# Defining a numerical method to simulate the STD given the estimated parameters


def euler_maruyama(dt, n_steps, alpha, mu, sigma):
    dW = np.random.normal(0, dt, n_steps)
    X = np.zeros(n_steps)
    X[0] = mu
    for i in range(1, n_steps):
        X[i] = X[i - 1] + alpha * (mu - X[i - 1]) * dt + sigma * dW[i]
    return X

# Defining a a function to calculate RMSE of predicted data compared to observed data


def root_mean_squared_error(y_obs, y_pred):
    squared_errors = (y_obs - y_pred) ** 2
    rmse = np.sqrt(np.mean(squared_errors))
    return rmse


# Choosing the model with the least RMSE out of 100000 trials
estimation_steps = 360
total_rmse = 0
trials = 100000
rmse_max = 0
rmse_min = 5
for _ in range(trials):
    X_estimated = euler_maruyama(
        dt, estimation_steps, alpha_mle, mu_mle, sigma_mle)
    X_simulated = X_estimated[:n_steps]
    rmse = root_mean_squared_error(stock_value, X_simulated)
    total_rmse = total_rmse+rmse
    if rmse > rmse_max:
        rmse_max = rmse
    if rmse < rmse_min:
        rmse_min = rmse
        best_simulation = X_simulated
        best_estimation = X_estimated

time_steps = np.arange(0, total_step_length, dt)

plt.plot(time_steps, stock_value,
         label='Observed Data', color='limegreen')
plt.plot(time_steps, best_simulation,
         label='Prediction', color='blue')

plt.title('Predictive model based on observed data')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.grid(True)
plt.show()

print("Average Root Mean Squared Error (RMSE):", total_rmse/trials)
print("Max:", rmse_max)
print("Min/Best RMSE:", root_mean_squared_error(stock_value, best_simulation))

# Plotting a graph for days beyond the observed data
estimation_time_steps = np.arange(0, estimation_steps, dt)
plt.plot(time_steps, stock_value,
         label='Observed Data', color='limegreen')
plt.plot(estimation_time_steps, best_estimation,
         label='Predicted Data', color='blue')

plt.title('Predictive model for future stock prices')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.grid(True)
plt.show()
