import numpy as np
import pandas as pd
import yfinance as yf
import datetime

def compute_returns_and_covariance(tickers, period="5y", n_years = None):

    monthly_closes = yf.download(tickers, period = period, interval = "1mo")['Close']

    yearly_closes = monthly_closes[::12]

    last_close = yf.download(tickers, start = datetime.date.today() - datetime.timedelta(days=1), end = datetime.date.today(), interval = "1mo")['Close']

    close_prices = pd.concat([yearly_closes, last_close])

    if n_years is not None:

        close_prices = close_prices[:n_years+1]

    log_returns_df = np.log(close_prices / close_prices.shift(1))

    log_returns = log_returns_df[1:].values.transpose()

    mean_returns = np.mean(log_returns, axis = 1)

    return mean_returns, np.cov(log_returns)


