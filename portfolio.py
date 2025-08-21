from amplpy import AMPL
import numpy as np
import pandas as pd
from utils import compute_returns_and_covariance
from scipy.optimize import minimize

class Portfolio:

    def get_returns_and_covariance(self, tickers, period, train_period):

        returns, sigma = compute_returns_and_covariance(tickers, period = period, train_period = train_period)

        sigma = pd.DataFrame(sigma, index=range(len(tickers)), columns=range(len(tickers)))

        return returns, sigma

    def get_ret_vol_sr(self, w, mean_return, sigma):

        ret = np.dot(mean_return, w)
        vol = np.sqrt(np.dot(w.T, np.dot(sigma, w)))
        sr = ret / vol

        return np.array([ret, vol, sr])

    def output_weights(self):

        portfolio_stats = self.get_ret_vol_sr(self.weights, self.returns, self.sigma)

        portfolio_stats = pd.DataFrame({
            'Return': [portfolio_stats[0]],
            'Volatility': [portfolio_stats[1]]
        })

        return portfolio_stats, self.weights

    def __init__(self, tickers, period, train_period):

        self.tickers = tickers
        self.returns, self.sigma = self.get_returns_and_covariance(tickers, period, train_period)

class StandardMarkowitz(Portfolio):

    def __init__(self, tickers, risk_aversion, diversification):

        super().__init__(tickers)

        self.portfolio = AMPL()
        self.portfolio.option["solver"] = "highs"
        self.portfolio.read('ampl_files/standard_markowitz.mod')
        self.portfolio.set["S"] = list(range(len(tickers)))
        self.portfolio.param["r"] = self.returns
        self.portfolio.param["Sigma"] = self.sigma
        self.portfolio.param["risk_aversion"] = risk_aversion
        self.portfolio.param["lower_divers"] = diversification[0]
        self.portfolio.param["upper_divers"] = diversification[1] if diversification[1] is not None else len(tickers)

        self.portfolio.solve()
        self.weights = list( self.portfolio.var["x"].to_dict().values() )
        self.weights = np.array(self.weights)

class PortfolioSimulation(Portfolio):

    def simulate_portfolios(self):

        n_portfolios = 10000

        weights_array = np.zeros( (n_portfolios, len(self.tickers)) )
        return_array = np.zeros(n_portfolios)
        volatility_array = np.zeros(n_portfolios)
        sharpe_array = np.zeros(n_portfolios)

        for i in range(n_portfolios):

            weights = np.random.random(len(self.tickers))

            weights = weights / np.sum(weights)

            weights_array[i, :] = weights

            return_array[i] = np.dot(self.returns, weights)

            volatility_array[i] = np.sqrt(np.dot(weights.T, np.dot(self.sigma, weights)))

            sharpe_array[i] = return_array[i] / volatility_array[i]

        portfolios_data = pd.DataFrame({
            'Return': return_array,
            'Volatility': volatility_array,
            'Sharpe': sharpe_array
        })

        return portfolios_data

    def sharpe_portfolio(self):

        function_to_minimize = lambda w: -self.get_ret_vol_sr(w, self.returns, self.sigma)[2]

        def check_sum(w):
            return np.sum(w) - 1

        bounds = tuple((0, 1) for _ in range(len(self.tickers)))
        init_guess = np.repeat(0.25, len(self.tickers))
        cons = ({'type': 'eq', 'fun': check_sum})
        opt_results = minimize(function_to_minimize, init_guess, method='SLSQP', bounds=bounds, constraints=cons)

        return opt_results.x

    def __init__(self, tickers):

        super().__init__(tickers)
        self.weights = self.sharpe_portfolio()

