import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf


def get_data(company_symbol, start_date, end_date, col_name=None):
    # Fetch historical data from Yahoo Finance
    data = yf.download(company_symbol, start=start_date, end=end_date, progress=False)

    # Filter data for Open, Close, and Adjusted Close columns
    # print(data.columns)
    # data = data[["Open", "Close", "Adj Close"]]
    # print(col_name)
    if col_name != None:
        data.rename(columns={col_name: company_symbol}, inplace=True)
    else:
        data.rename(columns={"Adj Close": company_symbol}, inplace=True)
    # print(data.columns)
    return data


# Plot the adjusted close price over time
def plotReturns(data, symbol):
    # data["Date"] = pd.to_datetime(data["Date"])
    # filtered_data = data.loc[(data["Date"] >= start_date) & (data["Date"] <= end_date)]
    plt.figure(figsize=(10, 6))
    data["Date"] = pd.to_datetime(data.index)
    plt.plot(
        data["Date"],
        data["Benchmark"],
        linestyle="-",
        color="blue",
        label="Benchmark",
    )
    plt.plot(
        data["Date"],
        data["RL"],
        linestyle="-",
        color="green",
        label="Q Trader",
    )
    plt.plot(
        data["Date"],
        data["QRL"],
        linestyle="-",
        color="red",
        label="QRL Trader",
    )
    plt.legend()
    plt.title("Portfolio Value Over Time")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("Figure1.png")


def get_stats(portvals):
    daily_returns = portvals.pct_change(1)
    daily_returns = daily_returns[1:]
    adr = np.mean(daily_returns)
    sddr = np.std(daily_returns)
    cr = 0.0
    final_value = portvals.iloc[0, 0]
    if len(portvals) > 1:
        cr = portvals.iloc[-1, 0] / portvals.iloc[0, 0] - 1
        final_value = portvals.iloc[-1, 0]
    print("Start Value " + str(portvals.iloc[0, 0]))
    sr = (np.sqrt(252) * adr) / sddr
    print("Sharpe Ratio " + str(sr))
    print("Cumulative Return " + str(cr))
    print("Mean Daily Returns " + str(adr))
    print("Standard Deviation of Daily Returns " + str(sddr))
    print("Date Range " + str(portvals.index[0]) + " " + str(portvals.index[-1]))
    print("Final Portfolio Value " + str(final_value))
