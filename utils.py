import numpy as np
import yfinance as yf

def compute_returns(tickers, period="5y"):

    close_prices = yf.download(tickers, period = period, interval = "3mo")['Close']

    n_years = int(list(period)[0])

    returns = np.zeros(shape = (len(tickers), n_years))

    for i in range(len(tickers)):
        ticker_close_prices = close_prices[tickers[i]].values

        annual_returns = []

        for j in range(n_years):

            p_0 = ticker_close_prices[4*j]
            p_T = ticker_close_prices[4*(j+1)]

            annual_returns.append( (p_T - p_0) / p_0 )

        returns[i,:] = annual_returns

    mean_returns = np.mean(returns, axis = 1)

    return mean_returns, returns

def compute_covariance(tickers, period="5y"):

    returns = compute_returns(tickers, period)[1]

    return np.cov(returns)
