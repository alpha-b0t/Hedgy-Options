import math
from scipy.stats import norm
import numpy as np
from enum import Enum

class OptionType(Enum):
    CALL = 0
    PUT = 1

class Models():
    @staticmethod
    def black_scholes(S, K, T, r, sigma, option_type: OptionType):
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)

        if option_type == OptionType.CALL:
            option_price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        else:
            option_price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        return option_price

    @staticmethod
    def binomial_option_pricing(S, K, T, r, sigma, n, option_type: OptionType):
        dt = T / n
        u = math.exp(sigma * math.sqrt(dt))
        d = 1 / u
        p = (math.exp(r * dt) - d) / (u - d)

        # Build binomial price tree
        stock_prices = [[S * u**j * d**(n-j) for j in range(i+1)] for i in range(n+1)]
        option_values = [[0 for j in range(i+1)] for i in range(n+1)]

        # Calculate option values at expiration
        if option_type == OptionType.CALL:
            for j in range(n+1):
                option_values[n][j] = max(0, stock_prices[n][j] - K)
        else:
            for j in range(n+1):
                option_values[n][j] = max(0, K - stock_prices[n][j])

        # Calculate option values at earlier nodes
        for i in range(n-1, -1, -1):
            for j in range(i+1):
                option_values[i][j] = math.exp(-r * dt) * (p * option_values[i+1][j] + (1 - p) * option_values[i+1][j+1])

        return option_values[0][0], option_values

    @staticmethod
    def monte_carlo(S, K, T, r, sigma, n, option_type: OptionType, steps=252):
        dt = T / steps
        S_simulations = np.zeros(n)
        payoff_sum = 0

        if option_type == OptionType.CALL:
            for i in range(n):
                z = np.random.normal(0, 1)
                S_simulations[i] = S * math.exp((r - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * z)

                payoff = max(0, S_simulations[i] - K)

                payoff_sum += payoff
        else:
            for i in range(n):
                z = np.random.normal(0, 1)
                S_simulations[i] = S * math.exp((r - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * z)

                payoff = max(0, K - S_simulations[i])

                payoff_sum += payoff

        option_price = math.exp(-r * T) * (payoff_sum / n)
        return option_price

if __name__ == '__main__':
    S = 140.88
    K = 65
    T = 456/252
    r = 0.05
    sigma = 0.3477
    n_binom = 1000
    n_monte_carlo = 100000
    steps = 252
    option_type = OptionType.CALL

    print(f"Black-Scholes: {Models.black_scholes(S, K, T, r, sigma, option_type)}")

    binom_options_value_at_expiration, binom_options_values = Models.binomial_option_pricing(S, K, T, r, sigma, n_binom, option_type)
    
    print(f"BINOM: {binom_options_value_at_expiration}")

    print(f"Monte Carlo: {Models.monte_carlo(S, K, T, r, sigma, n_monte_carlo, option_type, steps)}")
