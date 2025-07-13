from amplpy import AMPL
import numpy as np
import pandas as pd


class MarkowitzPortfolio:

    def get_returns(self, tickers):

        return np.array([1.2, 1.1, 1.3])

    def get_covariances(self, tickers):

        sigma = np.matrix([[1.5, 0.5, 2], [0.5, 2, 0], [2, 0, 5]])
        sigma = pd.DataFrame(sigma, index=range(len(tickers)), columns=range(len(tickers)))

        return sigma

    def __init__(self, tickers):

        self.returns = self.get_returns(tickers)
        self.sigma = self.get_covariances(tickers)
        self.portfolio = AMPL()
        self.portfolio.option["solver"] = "highs"
        self.portfolio.option["verbose"] = False

    def output_weights(self):
        self.portfolio.solve()
        print(self.portfolio.var["x"].to_dict())
        print(self.portfolio.var["x_risk_free"].value())


class DeterministicMarkowitz(MarkowitzPortfolio):

    def __init__(self, tickers, r_risk_free, risk_aversion, divers, file_path):

        super().__init__(tickers)

        self.portfolio.read(file_path)
        self.portfolio.set["S"] = list(range(len(tickers)))
        self.portfolio.param["r_risk_free"] = r_risk_free
        self.portfolio.param["r"] = self.returns
        self.portfolio.param["Sigma"] = self.sigma
        self.portfolio.param["risk_aversion"] = risk_aversion
        self.portfolio.param["lower_divers"] = divers[0]
        self.portfolio.param["upper_divers"] = divers[1] if divers[1] is not None else len(tickers)



DM = DeterministicMarkowitz(["AA", "AAPL", "TSLA"], 1.01, 1, [1,3], 'deterministic_markowitz.mod')
DM.output_weights()
