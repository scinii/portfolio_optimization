from portfolio import Portfolio, StandardMarkowitz, StochasticMarkowitz


def main():

    tickers = ["AAPL", "TSLA","AA","V","MA"]

    trial = StochasticMarkowitz(tickers = tickers, period = "10y", train_period = 5, alpha = 0.1, beta= 0.4)
    fig = trial.plot_efficient_frontier()
    fig.show()

if __name__ == "__main__":

    main()
