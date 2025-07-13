from amplpy import AMPL
import numpy as np
import pandas as pd


def deterministic_markowitz(tickers, r_risk_free, risk_aversion, divers = [0, None] ):

    """
    :param tickers: a list of tickers
    :param r_risk_free: return of the risk-free asset
    :param risk_aversion: coefficient that measures how risk-adverse the user is
    :param divers: list, its elements give and lower and upper bound (respectively) for the number of stocks
    :return: weights to assign to the stocks and risk-free asset
    """

    # some placeholder values to test while we wait for Job's function

    sigma = np.matrix([[1.5, 0.5, 2], [0.5, 2, 0], [2, 0, 5]])
    sigma = pd.DataFrame(sigma, index=range( len(tickers) ), columns=range( len(tickers) ))
    r = np.array([1.2, 1.1, 1.3])

    portfolio = AMPL()
    portfolio.read("deterministic_markowitz.mod")

    portfolio.set["S"] = list(range(3))

    portfolio.param["r_risk_free"] = r_risk_free
    portfolio.param["r"] = r
    portfolio.param["Sigma"] = sigma
    portfolio.param["risk_aversion"] = risk_aversion
    portfolio.param["lower_divers"] = divers[0]
    portfolio.param["upper_divers"] = divers[1] if divers[1] is not None else len(tickers)

    portfolio.option["solver"] = "highs"
    portfolio.option["verbose"] = False
    portfolio.solve()

    return portfolio


result = deterministic_markowitz( ["AA", "AAPL", "TSLA"], 1.01, 1, [0,3] )
print(result.var["x"].to_dict())
print(result.var["x_risk_free"].value())

