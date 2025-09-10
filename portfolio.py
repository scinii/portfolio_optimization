from amplpy import AMPL
import numpy as np
import pandas as pd
from utils import compute_returns_and_covariance, get_timeseries
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm

class Portfolio:

    def get_returns_and_covariance(self, tickers, period, train_period):

        #returns, sigma = compute_returns_and_covariance(tickers, period = period, train_period = train_period)

        #sigma = pd.DataFrame(sigma, index=range(len(tickers)), columns=range(len(tickers)))

        return np.array([1.25, 1.15, 1.35]), np.array([[1.5, 0.5, 2], [0.5, 2, 0], [2, 0, 5]])

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

    def visualize_portfolios(self):

        fig = plt.figure(figsize=(10, 6))
        gs = gridspec.GridSpec(2, 3, height_ratios=[2, 1], figure=fig)

        ax_scatter = fig.add_subplot(gs[0, :2])
        ax_text = fig.add_subplot(gs[0, 2])
        ax_timeseries = fig.add_subplot(gs[1, :])

        portfolios_data = self.simulate_portfolios()
        timeseries = get_timeseries(self.weights, self.tickers, self.period)

        portfolios_data.plot.scatter(x = 'Volatility', y = 'Return', c = 'Sharpe',
                                    colormap = 'viridis', ax = ax_scatter, colorbar=False)

        self.additional_portfolio.plot.scatter(x = 'Volatility', y = 'Return', c = 'r', ax = ax_scatter)

        sharpe_ratio = self.additional_portfolio['Sharpe'].values[0]
        period_return = timeseries.values[-1]-1

        timeseries.plot.line(y = 'Value', ax = ax_timeseries)

        ax_text.axis("off")  # turn off axes
        ax_text.text(
            0.05, 0.8, f"Sharpe ratio={sharpe_ratio:.2f}", ha="left", va="center", fontsize=12
        )
        ax_text.text(
            0.05, 0.6, f"Return={period_return:.2f}", ha="left", va="center", fontsize=12
        )

        plt.tight_layout()
        plt.show()

    def __init__(self, tickers, period, train_period):

        self.tickers = tickers
        self.period = period
        self.train_period = train_period

        self.returns, self.sigma = self.get_returns_and_covariance(tickers, self.period, self.train_period )

class StandardMarkowitz(Portfolio):

    def __init__(self, tickers, period, train_period, risk_aversion):

        super().__init__(tickers, period, train_period)

        self.portfolio = AMPL()
        self.portfolio.option["solver"] = "highs"
        self.portfolio.read('ampl_files/standard_markowitz.mod')
        self.portfolio.set["S"] = list(range(len(tickers)))
        self.portfolio.param["r"] = self.returns
        self.portfolio.param["Sigma"] = self.sigma
        self.portfolio.param["risk_aversion"] = risk_aversion

        self.portfolio.solve()
        self.weights = list(self.portfolio.var["x"].to_dict().values())
        self.weights = np.array(self.weights)
        self.additional_portfolio = self.output_weights()[0]

class StochasticMarkowitz(Portfolio):

    def __init__(self, tickers, period, train_period, alpha, beta):

        super().__init__(tickers, period, train_period)

        self.portfolio = AMPL()
        self.portfolio.option["solver"] = "highs"
        self.portfolio.read('ampl_files/stochastic_markowitz.mod')
        self.portfolio.set["S"] = list(range(len(tickers)))
        self.portfolio.param["r"] = self.returns
        self.portfolio.param["Sigma"] = self.sigma

        self.portfolio.param["alpha"] = alpha
        self.portfolio.param["phi"] = norm.ppf(1-beta)

        self.portfolio.solve()
        self.weights = list(self.portfolio.var["x"].to_dict().values())
        self.weights = np.array(self.weights)
        self.additional_portfolio = self.output_weights()[0]
