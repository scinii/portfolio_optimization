from amplpy import AMPL
import numpy as np
import pandas as pd
from utils import compute_returns_and_covariance, get_timeseries
import matplotlib.pyplot as plt
from scipy.stats import norm

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
            'Volatility': [portfolio_stats[1]],
            'Sharpe': [portfolio_stats[2]]
        })

        return portfolio_stats, self.weights

    def simulate_portfolios(self):

        n_portfolios = 50000

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

    def get_plot_timeseries(self):

        timeseries = get_timeseries(self.weights, self.tickers, self.period)

        fig, ax = plt.subplots(figsize=(5, 4))

        # Plotly-like style
        ax.plot(timeseries.index, timeseries.values, color='#1f77b4', linewidth=2)  # Plotly default blue
        ax.grid(True, linestyle='--', alpha=0.5)  # light dashed grid
        ax.set_facecolor('#e5ecf6')
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")

        for spine in ax.spines.values():
            spine.set_visible(False)


        return fig

    def get_plot_efficient_frontier(self):

        portfolios_data = self.simulate_portfolios()

        fig, ax = plt.subplots(figsize=(4, 4))

        portfolios_data.plot.scatter(x = 'Volatility', y = 'Return', c = 'Sharpe',colormap='viridis',ax=ax)

        self.your_portfolio.plot.scatter(x = 'Volatility', y = 'Return', color="red",s=25,ax=ax,
                                                 label='Your Portfolio', marker='*')

        return fig

    def __init__(self, tickers = None, period = "5y", train_period = 2):

        self.portfolio = AMPL()
        self.portfolio.option["solver"] = "highs"

        self.tickers = tickers
        self.period = period
        self.train_period = train_period

        self.weights = None
        self.your_portfolio = None

        self.returns, self.sigma = self.get_returns_and_covariance(tickers, self.period, self.train_period )

    def set_tickers(self, tickers):
        self.tickers = tickers

    def set_period(self, period):
        self.period = period

    def set_train_period(self, train_period):
        self.train_period = train_period

    def solve_portfolio(self):

        self.portfolio.solve()
        self.weights = list(self.portfolio.var["x"].to_dict().values())
        self.weights = np.array(self.weights)
        self.your_portfolio = self.output_weights()[0]

class StandardMarkowitz(Portfolio):

    def __init__(self, tickers = None, period = '5y', train_period = 2, risk_aversion = 1.0):

        super().__init__(tickers, period, train_period)

        self.risk_aversion = risk_aversion

        self.portfolio.read('ampl_files/standard_markowitz.mod')
        self.portfolio.set["S"] = list(range(len(tickers)))
        self.portfolio.param["r"] = self.returns
        self.portfolio.param["Sigma"] = self.sigma
        self.portfolio.param["risk_aversion"] = self.risk_aversion

    def set_risk_aversion(self, new_risk_aversion):

        self.risk_aversion = new_risk_aversion
        self.portfolio.param["risk_aversion"] = self.risk_aversion

class StochasticMarkowitz(Portfolio):

    def __init__(self, tickers = None, period = '5y', train_period = 2, alpha = 0.2, beta = 0.5):

        super().__init__(tickers, period, train_period)

        self.alpha = alpha
        self.beta = beta

        self.portfolio.read('ampl_files/stochastic_markowitz.mod')
        self.portfolio.set["S"] = list(range(len(tickers)))
        self.portfolio.param["r"] = self.returns
        self.portfolio.param["Sigma"] = self.sigma

        self.portfolio.param["alpha"] = alpha
        self.portfolio.param["phi"] = norm.ppf(1-beta)

    def set_alpha(self,new_alpha):

        self.alpha = new_alpha
        self.portfolio.param["alpha"] = self.alpha

    def set_beta(self,new_beta):

        self.beta = new_beta
        self.portfolio.param["beta"] = self.beta



