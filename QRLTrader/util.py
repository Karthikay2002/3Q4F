import matplotlib.pyplot as plt
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
        color="blue",
        label="Q Trader",
    )
    plt.plot(
        data["Date"],
        data["QRL"],
        linestyle="-",
        color="blue",
        label="QRL Trader",
    )
    plt.title("Portfolio Value Over Time")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("Figure1.png")
