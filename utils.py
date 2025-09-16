import numpy as np
import pandas as pd
import yfinance as yf

def compute_returns_and_covariance(tickers, period="10y", train_period = None):

    monthly_closes = yf.download(tickers, period = period, interval = "1mo", auto_adjust=True)['Close']

    yearly_closes = monthly_closes[::12]

    close_prices = pd.concat([ yearly_closes, monthly_closes.iloc[[-1]] ] )

    if train_period is not None:

        close_prices = close_prices[:train_period+1]

    log_returns_df = np.log(close_prices / close_prices.shift(1))

    log_returns = log_returns_df[1:].values.transpose()

    mean_returns = np.mean(log_returns, axis = 1)

    return mean_returns, np.cov(log_returns)

def get_timeseries(w, tickers, period):

    daily_closes = yf.download(tickers, period=period, interval="1d", auto_adjust=True)['Close']
    price_on_start_date = daily_closes.iloc[0]
    timeseries = (daily_closes * w / price_on_start_date).sum(axis=1)

    return timeseries


