from portfolio import Portfolio, StandardMarkowitz, StochasticMarkowitz


def main():

    tickers = ["AAPL", "TSLA","AA","V","MA"]

    trial = StochasticMarkowitz(tickers, "10y", 5, 0.1,0.2)
    trial.visualize_portfolios()

if __name__ == "__main__":

    main()
