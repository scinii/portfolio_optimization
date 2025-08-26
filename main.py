from portfolio import Portfolio, StandardMarkowitz


def main():

    tickers = ["AAPL", "TSLA","AA"]
    risk_aversion = 0.1
    diversification = [0,3]

    trial = StandardMarkowitz(tickers, "10y", 5, risk_aversion, diversification)
    trial.visualize_portfolios()

if __name__ == "__main__":

    main()
