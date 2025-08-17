"""
This file will find the expected returns (r) and covariance matrixes (sigma) for stock tickers from yahoofinance.
"""
import yfinance as yf


def compute_expected_return_and_covariance(tickers, period="1y", interval="1d"):
    """
    parameter meanings:
    tickers: list of stock symbols, for example ['AAPL', 'TSLA']
    period: download period, for example '1y'
    interval: data frequency, for example '1d'
    function output:
    r: list of average daily returns, same order as tickers
    sigma: covariance matrix
    """

    #get close prices
    data = yf.download(tickers, period=period, interval=interval)
    price_data = data['Adj Close']

    #Find daily returns
    returns = {}
    for t in tickers:
        prices = price_data[t].tolist()
        daily = []
        for i in range(1, len(prices)):
            prev, curr = prices[i-1], prices[i]
            if prev == 0:
                daily.append(0)
            else:
                daily.append((curr - prev)/prev)
        returns[t] = daily

    #expected returns
    r = []
    for t in tickers:
        vals = returns[t]
        if vals:
            r.append(sum(vals) / len(vals))
        else:
            r.append(0)

    #Get covariance matrix sigma
    n = len(tickers)
    sigma = []
    for i in range(n):
        row = []
        ri = returns[tickers[i]]
        for j in range(n):
            rj = returns[tickers[j]]
            m_i, m_j = r[i], r[j]
            length = min(len(ri), len(rj))
            if length < 2:
                row.append(0)
            else:
                s = 0
                for k in range(length):
                    s += (ri[k] - m_i) * (rj[k] - m_j)
                row.append(s / (length - 1))
        sigma.append(row)

    return r, sigma