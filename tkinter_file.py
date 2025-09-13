from portfolio import StandardMarkowitz, StochasticMarkowitz
from tkinter import *
import matplotlib


root = Tk()
root.title("Portfolio Optimization")
root.iconbitmap('img/git_logo.ico')
root.geometry("500x500")


ticker_label = Label(root, text="Enter tickers (comma separated):")
ticker_label.pack(pady=5)

ticker_entry = Entry(root, width=40)
ticker_entry.pack(pady=5)
ticker_entry.insert(0, "AAPL, TSLA, AA, V, MA")   # default text

matplotlib.use("TkAgg")

def run_simulation():

    tickers = [t.strip() for t in ticker_entry.get().split(",")]
    trial = StochasticMarkowitz(tickers=tickers, period="10y", train_period=5, alpha=0.1, beta=0.4)
    timeseries = trial.get_plot_timeseries()
    scatter_plot = trial.get_plot_efficient_frontier()
    scatter_plot.show()
    timeseries.show()


run_button = Button(root, text="Run", command= run_simulation)
run_button.pack(pady=10)

root.mainloop()