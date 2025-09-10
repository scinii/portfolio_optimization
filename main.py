from portfolio import Portfolio, StandardMarkowitz, StochasticMarkowitz


def main():

    tickers = ["AAPL", "TSLA","AA"]
    risk_aversion = 1.05

    trial = StochasticMarkowitz(tickers, "5y", 5, 0.6, 0.3)
    #trial.visualize_portfolios()

if __name__ == "__main__":

    main()
