from amplpy import AMPL
import numpy as np
import pandas as pd
from tabulate import tabulate
from utils import compute_returns, compute_covariance

class MarkowitzPortfolio:

    def get_returns(self, tickers):

        return compute_returns(tickers)[0]

    def get_covariances(self, tickers):

        sigma = compute_covariance(tickers)
        sigma = pd.DataFrame(sigma, index=range(len(tickers)), columns=range(len(tickers)))

        return sigma

    def __init__(self, tickers):

        self.tickers = tickers
        self.returns = self.get_returns(tickers)
        self.sigma = self.get_covariances(tickers)
        self.portfolio = AMPL()

    def output_weights(self):

        self.portfolio.solve()

        risk_free_weight = self.portfolio.var["x_risk_free"].value()
        risky_assets_weights = list(self.portfolio.var["x"].to_dict().values())

        weights = risky_assets_weights + [risk_free_weight]
        asset_names = self.tickers + ["Bond"]

        headers = ["Asset", "Weight"]
        final_table = np.column_stack((asset_names, weights))

        print(tabulate(final_table, headers=headers, tablefmt="fancy_grid"))

class StandardMarkowitz(MarkowitzPortfolio):

    def __init__(self, tickers, r_risk_free, risk_aversion, divers):

        super().__init__(tickers)
        self.portfolio.option["solver"] = "highs"
        self.portfolio.read('ampl_files/standard_markowitz.mod')
        self.portfolio.set["S"] = list(range(len(tickers)))
        self.portfolio.param["r_risk_free"] = r_risk_free
        self.portfolio.param["r"] = self.returns
        self.portfolio.param["Sigma"] = self.sigma
        self.portfolio.param["risk_aversion"] = risk_aversion
        self.portfolio.param["lower_divers"] = divers[0]
        self.portfolio.param["upper_divers"] = divers[1] if divers[1] is not None else len(tickers)

class ConicMarkowitz(MarkowitzPortfolio):

    def __init__(self, tickers, r_risk_free, risk_tolerance, divers):

        super().__init__(tickers)

        self.portfolio.option["solver"] = "mosek"
        self.portfolio.read('ampl_files/conic_markowitz.mod')
        self.portfolio.set["S"] = list(range(len(tickers)))
        self.portfolio.param["r_risk_free"] = r_risk_free
        self.portfolio.param["r"] = self.returns
        self.portfolio.param["Sigma"] = self.sigma
        self.portfolio.param["risk_tolerance"] = risk_tolerance
        self.portfolio.param["lower_divers"] = divers[0]
        self.portfolio.param["upper_divers"] = divers[1] if divers[1] is not None else len(tickers)

class StochasticMarkowitz(MarkowitzPortfolio):

    def __init__(self, tickers, r_risk_free, risk_tolerance, divers):

        super().__init__(tickers)

        self.portfolio.option["solver"] = "highs"
        self.portfolio.read('ampl_files/stochastic_markowitz.mod')
        self.portfolio.set["S"] = list(range(len(tickers)))
        self.portfolio.param["r_risk_free"] = r_risk_free
        self.portfolio.param["r"] = self.returns
        self.portfolio.param["Sigma"] = self.sigma
        self.portfolio.param["risk_tolerance"] = risk_tolerance
        self.portfolio.param["lower_divers"] = divers[0]
        self.portfolio.param["upper_divers"] = divers[1] if divers[1] is not None else len(tickers)


DM = StandardMarkowitz(["AAPL", "TSLA"], 1.05, 0.1, [0,3])
DM.output_weights()
