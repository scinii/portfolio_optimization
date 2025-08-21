import numpy as np
import pandas as pd
import yfinance as yf
import datetime

import hvplot.pandas  # noqa
import panel as pn
import holoviews as hv
hv.extension('bokeh')
pn.extension('tabulator', design='material', template='material', theme_toggle=True, loading_indicator=True)

def compute_returns_and_covariance(tickers, period="10y", train_period = None):

    monthly_closes = yf.download(tickers, period = period, interval = "1mo", auto_adjust=True)['Close']

    yearly_closes = monthly_closes[::12]

    last_close = yf.download(tickers, start = datetime.date.today() - datetime.timedelta(days=1), end = datetime.date.today(), interval = "1d", auto_adjust=True)['Close']

    close_prices = pd.concat([yearly_closes, last_close])

    if train_period is not None:

        close_prices = close_prices[:train_period+1]

    log_returns_df = np.log(close_prices / close_prices.shift(1))

    log_returns = log_returns_df[1:].values.transpose()

    mean_returns = np.mean(log_returns, axis = 1)

    return mean_returns, np.cov(log_returns)

def plot_performance(tickers, portfolio_weights,theoretical_portfolio, period="10y"):

    daily_closes = yf.download(tickers, period=period, interval="1d", auto_adjust=True)['Close']

    def compute_timeseries(w, daily_closes):

        price_on_start_date = daily_closes.iloc[0]
        timeseries = (daily_closes * w / price_on_start_date).sum(axis=1)

        return timeseries

    yours_timeseries = compute_timeseries(portfolio_weights)

    plot = yours_timeseries.hvplot.line(
        ylabel="Total Value ($)",
        title="Portfolio performance",
        color = "green",
        responsive=True,
        min_height=400)

    hvplot.show(plot)

def visualize_portfolio(tickers, additional_portofolio):

    sigma = compute_returns_and_covariance(tickers)[1]
    portfolios_data = portfolio_simulation(tickers)
    optimal_portfolio = best_portfolio(tickers)

    scatter = portfolios_data.hvplot.scatter(x='Volatility', y='Return', c='Sharpe', cmap='plasma', width=600, height=400, colorbar=True, padding=0.1)
    best = optimal_portfolio.hvplot.scatter(x='Volatility', y='Return', color='red', line_color='black', s=25, )
    yours = additional_portofolio.hvplot.scatter(x='Volatility', y='Return', color='green', line_color='black', s=25, )
    whole_picture = scatter * best * yours

    summary = pn.pane.Markdown(
             f"""
        The selected portfolio has a volatility of {additional_portofolio.Volatility[0]:.2f}, a return of {additional_portofolio.Return[0]:.2f}
        and Sharpe ratio of {additional_portofolio.Return[0] / additional_portofolio.Volatility[0]:.2f}.""",
        width=350
    )

    final = pn.Row(whole_picture, pn.Column(summary), sizing_mode="stretch_both")

    sigma = pd.DataFrame(sigma, index=tickers, columns=tickers)

    annual_log_ret_cov_heatmap = sigma.hvplot.heatmap(cmap='viridis', title='Log Return Covariance')

    hvplot.show(annual_log_ret_cov_heatmap)


    return True

