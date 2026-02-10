import math
from scipy.stats import norm
import numpy as np

class OptionPricingModels():
    @staticmethod
    def black_scholes(S, K, T, r, sigma, q, option_type):
        # Time complexity: O(1)
        # Space complexity: O(1)
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)

        if option_type == "call":
            option_price = S * math.exp(-q * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        else:
            option_price = K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-q * T) * norm.cdf(-d1)

        return option_price
    
    @staticmethod
    def european_bopm(S, K, T, r, sigma, q, n, option_type):
        # Time complexity: O(n^2)
        # Space complexity: O(n^2)
        dt = T / n
        u = math.exp(sigma * math.sqrt(dt))
        d = 1 / u
        p = (math.exp((r-q) * dt) - d) / (u - d)

        # Create binomial price tree for the underlying asset
        stock_prices = [[0.0 for j in range(i+1)] for i in range(n+1)]

        for i in range(n+1):
            for j in range(i+1):
                stock_prices[i][j] = S * (u**(i-j)) * (d ** j)
        
        # Create option price tree
        option_prices = [[0.0 for j in range(i+1)] for i in range(n+1)]

        # Calculate option prices at expiration
        if option_type == "call":
            for j in range(n+1):
                option_prices[n][j] = max(0, stock_prices[n][j] - K)
        else:
            for j in range(n+1):
                option_prices[n][j] = max(0, K - stock_prices[n][j])

        # Calculate option prices at each node
        for i in range(n-1, -1, -1):
            for j in range(i+1):
                option_prices[i][j] = math.exp(-r * dt) * (p * option_prices[i+1][j] + (1 - p) * option_prices[i+1][j+1])

        return option_prices[0][0], option_prices, stock_prices

    @staticmethod
    def american_bopm(S, K, T, r, sigma, q, n, option_type):
        # Time complexity: O(n^2)
        # Space complexity: O(n^2)
        dt = T / n
        u = math.exp(sigma * math.sqrt(dt))
        d = 1 / u
        p = (math.exp((r-q) * dt) - d) / (u - d)

        # Create binomial price tree for the underlying asset
        stock_prices = [[0.0 for j in range(i+1)] for i in range(n+1)]

        for i in range(n+1):
            for j in range(i+1):
                stock_prices[i][j] = S * (u**(i-j)) * (d ** j)
        
        # Create option price tree
        option_prices = [[0.0 for j in range(i+1)] for i in range(n+1)]

        # Calculate option prices at expiration
        if option_type == "call":
            for j in range(n+1):
                option_prices[n][j] = max(0, stock_prices[n][j] - K)
        else:
            for j in range(n+1):
                option_prices[n][j] = max(0, K - stock_prices[n][j])

        # Calculate option prices at each node
        if option_type == "call":
            for i in range(n-1, -1, -1):
                for j in range(i+1):
                    option_prices[i][j] = math.exp(-r * dt) * (p * option_prices[i+1][j] + (1 - p) * option_prices[i+1][j+1])

                    # Check for early exercise
                    option_prices[i][j] = max(option_prices[i][j], stock_prices[i][j] - K)
        else:
            for i in range(n-1, -1, -1):
                for j in range(i+1):
                    option_prices[i][j] = math.exp(-r * dt) * (p * option_prices[i+1][j] + (1 - p) * option_prices[i+1][j+1])

                    # Check for early exercise
                    option_prices[i][j] = max(option_prices[i][j], K - stock_prices[i][j])

        return option_prices[0][0], option_prices, stock_prices

    @staticmethod
    def monte_carlo(S, K, T, r, sigma, n, option_type, steps=252):
        dt = T / steps
        S_simulations = np.zeros(n)
        payoff_sum = 0

        if option_type == "call":
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
    S = 0.7003
    K = 2.50
    T = 36/252
    r = 0.05
    sigma = 4.8036
    q = 0
    n_binom = 1000
    n_monte_carlo = 100000
    steps = 252
    option_type = "put"

    print(f"Black-Scholes: {OptionPricingModels.black_scholes(S, K, T, r, sigma, q, option_type)}")

    euro_bopm_fair_price, euro_bopm_options_prices, euro_bopm_stock_prices = OptionPricingModels.european_bopm(S, K, T, r, sigma, q, n_binom, option_type)

    print(f"European BOPM (fair value): {euro_bopm_fair_price}, stock price: {euro_bopm_stock_prices[0][0]}")
    
    american_bopm_fair_price, american_bopm_options_prices, american_bopm_stock_prices = OptionPricingModels.american_bopm(S, K, T, r, sigma, q, n_binom, option_type)
    
    print(f"American BOPM (fair value): {american_bopm_fair_price}, stock price: {american_bopm_stock_prices[0][0]}")
    print(f"Monte Carlo: {OptionPricingModels.monte_carlo(S, K, T, r, sigma, n_monte_carlo, option_type, steps)}")
