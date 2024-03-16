import math
from scipy.stats import norm
import numpy as np

def black_scholes(S, K, T, r, sigma, is_call=True):
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if is_call:
        option_price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        option_price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return option_price

def binomial_tree(S, K, T, r, sigma, n, is_call=True):
    dt = T / n
    u = math.exp(sigma * math.sqrt(dt))
    d = 1 / u
    p = (math.exp(r * dt) - d) / (u - d)

    # Build binomial price tree
    stock_prices = [[S * u**j * d**(n-j) for j in range(i+1)] for i in range(n+1)]
    option_values = [[0 for j in range(i+1)] for i in range(n+1)]

    # Calculate option values at expiration
    for j in range(n+1):
        if is_call:
            option_values[n][j] = max(0, stock_prices[n][j] - K)
        else:
            option_values[n][j] = max(0, K - stock_prices[n][j])

    # Calculate option values at earlier nodes
    for i in range(n-1, -1, -1):
        for j in range(i+1):
            option_values[i][j] = math.exp(-r * dt) * (p * option_values[i+1][j] + (1 - p) * option_values[i+1][j+1])

    return option_values[0][0]

def monte_carlo(S, K, T, r, sigma, num_simulations, is_call=True):
    dt = T / 252 # Is this correct?
    S_simulations = np.zeros(num_simulations)
    payoff_sum = 0

    for i in range(num_simulations):
        z = np.random.normal(0, 1)
        S_simulations[i] = S * math.exp((r - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * z)

        if is_call:
            payoff = max(0, S_simulations[i] - K)
        else:
            payoff = max(0, K - S_simulations[i])

        payoff_sum += payoff

    option_price = math.exp(-r * T) * (payoff_sum / num_simulations)
    return option_price
