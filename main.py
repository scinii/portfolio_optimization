from portfolio import Portfolio, StandardMarkowitz, StochasticMarkowitz


def main():

    tickers = ["AAPL", "TSLA","AA","V","MA"]

    trial = StandardMarkowitz(tickers = tickers, period = "10y", train_period = 5,  risk_aversion = 0.05 )
    trial.solve_portfolio()
    time = trial.get_plot_timeseries()
    time.show()

    scatter = trial.get_plot_efficient_frontier()
    scatter.show()

if __name__ == "__main__":

    main()
