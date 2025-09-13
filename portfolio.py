from amplpy import AMPL
import numpy as np
import pandas as pd
from utils import compute_returns_and_covariance, get_timeseries
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm

import plotly.express as px

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

        # fig = px.line(x=timeseries.index, y=timeseries.values,
        #               labels = {
        #                 "x" : "Date",
        #                 "y" : "Value"
        #               })

        return fig


    def get_plot_efficient_frontier(self):

        portfolios_data = self.simulate_portfolios()

        fig, ax = plt.subplots(figsize=(4, 4))

        portfolios_data.plot.scatter(x = 'Volatility', y = 'Return', c = 'Sharpe',colormap='viridis',ax=ax)

        (self.additional_portfolio).plot.scatter(x = 'Volatility', y = 'Return', color="red",s=25,ax=ax,
                                                 label='Your Portfolio', marker='*')

        # fig = px.scatter(portfolios_data, x="Volatility", y="Return", color = "Sharpe")
        # fig.add_scatter(x = self.additional_portfolio['Volatility'],
        #                 y = self.additional_portfolio['Return'],
        #                 marker = dict(color="red", size = 10), name = 'Portfolio', marker_symbol='star')
        #
        # fig.update_layout(legend=dict(
        #     yanchor="top",
        #     y=0.99,
        #     xanchor="left",
        #     x=0.01
        # ))

        return fig

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
