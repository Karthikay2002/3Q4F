import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

import util


def get_lookback_data(sd, ed, symbol, lookback, column="Adj Close"):
    nsd = sd - BDay(3 * lookback)
    price = util.get_data(symbol, sd, ed, col_name=column)
    # print(price.columns)
    price = price[[symbol]]
    return price


def getDailyRets(price):
    daily_rets = price.copy()
    daily_rets[1:] = (price[1:]) / (price[:-1].values) - 1.0
    daily_rets.ix[0, :] = 0
    return daily_rets


def getBollingerBandsIndicator(sd, ed, symbol, lookback=20, lower_t=0, upper_t=1):
    price = get_lookback_data(sd, ed, symbol, lookback)
    sma = price.rolling(window=lookback, min_periods=lookback).mean()
    rolling_std = price.rolling(window=lookback, min_periods=lookback).std()
    top_band = sma + (2 * rolling_std)
    bottom_band = sma - (2 * rolling_std)
    bbp = (price - bottom_band) / (top_band - bottom_band)
    bbp1 = bbp.copy()
    for col in bbp1.columns:
        bbp1[col] = 0
    bbp1[bbp < 0] = 1
    bbp1[bbp > 1] = -1
    return bbp1[sd:]


def getRSI(sd, ed, symbol, lookback=14):
    price = get_lookback_data(sd, ed, symbol, lookback)
    daily_rets = getDailyRets(price)
    up_rets = daily_rets[daily_rets >= 0].fillna(0).cumsum()
    down_rets = -1 * daily_rets[daily_rets < 0].fillna(0).cumsum()
    up_gain = price.copy()
    up_gain.ix[:, :] = 0
    up_gain.values[lookback:, :] = (
        up_rets.values[lookback:, :] - up_rets.values[:-lookback, :]
    )

    down_loss = price.copy()
    down_loss.ix[:, :] = 0
    down_loss.values[lookback:, :] = (
        down_rets.values[lookback:, :] - down_rets.values[:-lookback, :]
    )

    rs = (up_gain / lookback) / (down_loss / lookback)
    rsi = 100 - (100 / (1 + rs))
    rsi.ix[:lookback, :] = np.nan
    rsi[rsi == np.inf] = 100.0
    rsi1 = rsi.copy()
    for col in rsi.columns:
        rsi1[col] = 0
    rsi1[rsi < 30] = 1
    rsi1[rsi > 70] = -1
    return rsi1[sd:]


def getCCI(sd, ed, symbol, lookback=20):
    price = get_lookback_data(sd, ed, symbol, lookback)
    price_unadj = get_lookback_data(sd, ed, symbol, lookback, column="Close")
    high = get_lookback_data(sd, ed, symbol, lookback, column="High")
    low = get_lookback_data(sd, ed, symbol, lookback, column="Low")
    high = high * price / price_unadj
    low = low * price / price_unadj
    typical = (price + high + low) / 3.0
    sma_typical = typical.rolling(window=lookback).mean()
    mean_deviation = price.copy()
    mean_deviation.ix[:, :] = 1.0
    dates = list(price.index)
    for date in price[sd:].index:
        val = 0.0
        sid = dates.index(date)
        for i in range(lookback):
            nd = dates[sid - i]
            val = val + np.abs(sma_typical[symbol][date] - typical[symbol][nd])
        val = val / 20.0
        mean_deviation[symbol][date] = val
    cci = (typical - sma_typical) / (0.015 * mean_deviation)
    return cci[sd:]


def getEMAIndicator(sd, ed, symbol, lookback=20, lower_t=0.9, upper_t=1.1):
    price = get_lookback_data(sd, ed, symbol, lookback)
    ema = price.ewm(span=lookback, adjust=False).mean()
    ema = ema / price
    ema1 = ema.copy()
    for col in ema1.columns:
        ema1[col] = 0
    ema1[ema > 1.01] = 1
    ema1[ema < 0.99] = -1
    # print(ema1[symbol].value_counts())
    return ema1[sd:]


def getStochasticOscillatorIndicator(
    sd, ed, symbol, lookback=14, lower_t=20, upper_t=80
):
    price = get_lookback_data(sd, ed, symbol, lookback)
    price_unadj = get_lookback_data(sd, ed, symbol, lookback, column="Close")
    high = get_lookback_data(sd, ed, symbol, lookback, column="High")
    low = get_lookback_data(sd, ed, symbol, lookback, column="Low")
    high = high * price / price_unadj
    low = low * price / price_unadj
    min_low = low.rolling(window=lookback).min()
    max_high = high.rolling(window=lookback).max()
    K = (price - min_low) / (max_high - min_low)
    SK = K.rolling(window=3).mean()
    SD = SK.rolling(window=3).mean()
    SD1 = SD.copy()
    for col in SD1.columns:
        SD1[col] = 0
    SD1[SD < 0.2] = 1
    SD1[SD > 0.8] = -1
    # print(SD1[symbol].value_counts())
    return SD1[sd:]


def getBBPercentCharts(sd, ed, symbol="JPM"):
    bb, price, topband, bottomband = getBollingerBandsPercent(sd, ed, symbol)
    bb["Date"] = pd.to_datetime(bb.index)
    topband["Date"] = pd.to_datetime(bb.index)
    bottomband["Date"] = pd.to_datetime(bb.index)
    price["Date"] = pd.to_datetime(price.index)
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(24, 15), sharex=True)
    ax.plot(bb.index, bb[symbol])

    ax.set_ylabel("%B  Indicator", fontsize=20)
    ax.fill_between(
        bb["Date"], bb[symbol], 1, where=(bb[symbol] > 1), color="red", alpha=0.5
    )
    ax.fill_between(
        bb["Date"], bb[symbol], 0, where=(bb[symbol] < 0), color="green", alpha=0.5
    )
    ax.hlines(
        y=1,
        xmin=bb["Date"].iloc[0],
        xmax=bb["Date"].iloc[-1],
        color="red",
        linewidth=1,
        linestyles="--",
        label="Sell Signal",
    )
    ax.hlines(
        y=0,
        xmin=bb["Date"].iloc[0],
        xmax=bb["Date"].iloc[-1],
        color="green",
        linewidth=1,
        linestyles="--",
        label="Buy Signal",
    )
    ax.set_title("%B Indicator", fontsize=20)

    ax2.plot(price["Date"], price[symbol])
    ax2.plot(topband["Date"], topband[symbol], label="Top Band")
    ax2.plot(bottomband["Date"], bottomband[symbol], label="Bottom Band")
    ax2.legend(fontsize=20)
    ax2.set_ylabel("Normalized Price", fontsize=20)
    ax2.set_title("Stock Price With Bollinger Bands", fontsize=20)

    unique_months = bb["Date"].dt.strftime("%b %Y").unique()
    x_ticks = [
        bb[bb["Date"].dt.strftime("%b %Y") == month]["Date"].iloc[0]
        for month in unique_months
    ]
    x_labels = [month for month in unique_months]
    plt.xticks(x_ticks, x_labels, rotation=45, fontsize=20)
    plt.grid(True)
    plt.xlabel("Date", fontsize=20)
    plt.savefig("Figure2.png")


def getRSICharts(sd, ed, symbol="JPM"):
    rsi, price = getRSI(sd, ed, symbol)
    price = price / price.iloc[0]
    rsi["Date"] = pd.to_datetime(rsi.index)
    price["Date"] = pd.to_datetime(price.index)
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(24, 15), sharex=True)
    ax.plot(rsi.index, rsi[symbol])

    ax.fill_between(
        rsi["Date"],
        rsi[symbol],
        70,
        where=(rsi[symbol] > 70),
        color="red",
        alpha=0.5,
        label="Sell Signal",
    )
    ax.fill_between(
        rsi["Date"],
        rsi[symbol],
        30,
        where=(rsi[symbol] < 30),
        color="green",
        alpha=0.5,
        label="Buy Signal",
    )
    ax.hlines(
        y=70,
        xmin=rsi["Date"].iloc[0],
        xmax=rsi["Date"].iloc[-1],
        color="red",
        linewidth=1,
        linestyles="--",
    )
    ax.hlines(
        y=30,
        xmin=rsi["Date"].iloc[0],
        xmax=rsi["Date"].iloc[-1],
        color="green",
        linewidth=1,
        linestyles="--",
    )
    ax.set_title("Relative Strength Index", fontsize=20)
    ax.set_ylabel("RSI", fontsize=20)
    ax.legend(fontsize=20)

    ax2.plot(price["Date"], price[symbol])
    ax2.set_ylabel("Normalized Price", fontsize=20)

    unique_months = rsi["Date"].dt.strftime("%b %Y").unique()
    x_ticks = [
        rsi[rsi["Date"].dt.strftime("%b %Y") == month]["Date"].iloc[0]
        for month in unique_months
    ]
    x_labels = [month for month in unique_months]
    plt.xticks(x_ticks, x_labels, rotation=45, fontsize=20)
    plt.grid(True)
    plt.xlabel("Date")

    plt.savefig("Figure3.png")


def getEMACharts(sd, ed, symbol="JPM"):
    ema, price = getEMA(sd, ed, symbol)
    ema["Date"] = pd.to_datetime(ema.index)
    price["Date"] = pd.to_datetime(price.index)
    fig, ax = plt.subplots(figsize=(24, 7))
    ax.plot(price["Date"], price[symbol], label="Price")
    ax.plot(ema["Date"], ema[symbol], label="EMA")
    ax.fill_between(
        ema["Date"],
        ema[symbol],
        price[symbol],
        where=(ema[symbol] > price[symbol]),
        color="red",
        alpha=0.5,
        label="Sell Signal",
    )
    ax.fill_between(
        ema["Date"],
        ema[symbol],
        price[symbol],
        where=(ema[symbol] < price[symbol]),
        color="green",
        alpha=0.5,
        label="Buy Signal",
    )
    ax.legend(fontsize=20)
    plt.title("Exponential Moving Average with Stock Price", fontsize=20)
    unique_months = ema["Date"].dt.strftime("%b %Y").unique()
    x_ticks = [
        ema[ema["Date"].dt.strftime("%b %Y") == month]["Date"].iloc[0]
        for month in unique_months
    ]
    x_labels = [month for month in unique_months]
    plt.xticks(x_ticks, x_labels, rotation=45, fontsize=20)
    plt.grid(True)
    plt.xlabel("Date", fontsize=20)
    plt.ylabel("Price", fontsize=20)
    plt.subplots_adjust(top=0.95, bottom=0.20)
    plt.savefig("Figure4.png")


def getCCICharts(sd, ed, symbol):
    cci, price = getCCI(sd, ed, symbol)
    price = price / price.iloc[0]
    cci["Date"] = pd.to_datetime(cci.index)
    price["Date"] = pd.to_datetime(price.index)
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(24, 15), sharex=True)
    ax.plot(cci.index, cci[symbol])

    ax.set_ylabel("CCI", fontsize=20)
    ax.fill_between(
        cci["Date"],
        cci[symbol],
        100,
        where=(cci[symbol] > 100),
        color="red",
        alpha=0.5,
        label="Sell Signal",
    )
    ax.fill_between(
        cci["Date"],
        cci[symbol],
        -100,
        where=(cci[symbol] < -100),
        color="green",
        alpha=0.5,
        label="Buy Signal",
    )
    ax.hlines(
        y=100,
        xmin=cci["Date"].iloc[0],
        xmax=cci["Date"].iloc[-1],
        color="red",
        linewidth=1,
        linestyles="--",
    )
    ax.hlines(
        y=-100,
        xmin=cci["Date"].iloc[0],
        xmax=cci["Date"].iloc[-1],
        color="green",
        linewidth=1,
        linestyles="--",
    )
    ax.set_title("Commodity Channel Index", fontsize=20)
    ax.legend(fontsize=20)

    ax2.plot(price["Date"], price[symbol])
    ax2.set_ylabel("Normalized Price", fontsize=20)

    unique_months = cci["Date"].dt.strftime("%b %Y").unique()
    x_ticks = [
        cci[cci["Date"].dt.strftime("%b %Y") == month]["Date"].iloc[0]
        for month in unique_months
    ]
    x_labels = [month for month in unique_months]
    plt.xticks(x_ticks, x_labels, rotation=45, fontsize=20)
    plt.grid(True)
    plt.xlabel("Date", fontsize=20)
    plt.savefig("Figure5.png")


def getStochasticIndicatorCharts(sd, ed, symbol):
    sso, price = getStochasticOscillator(sd, ed, symbol)
    sso["Date"] = pd.to_datetime(sso.index)
    price = price / price.iloc[0]
    price["Date"] = pd.to_datetime(price.index)
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(24, 15), sharex=True)
    ax.plot(sso.index, sso[symbol])

    ax.set_ylabel("Stochastic Indicator", fontsize=20)
    ax.fill_between(
        sso["Date"],
        sso[symbol],
        80,
        where=(sso[symbol] > 80),
        color="red",
        alpha=0.5,
        label="Sell Signal",
    )
    ax.fill_between(
        sso["Date"],
        sso[symbol],
        20,
        where=(sso[symbol] < 20),
        color="green",
        alpha=0.5,
        label="Buy Signal",
    )
    ax.hlines(
        y=80,
        xmin=sso["Date"].iloc[0],
        xmax=sso["Date"].iloc[-1],
        color="red",
        linewidth=1,
        linestyles="--",
    )
    ax.hlines(
        y=20,
        xmin=sso["Date"].iloc[0],
        xmax=sso["Date"].iloc[-1],
        color="green",
        linewidth=1,
        linestyles="--",
    )
    ax.set_title("Slow Stochastic Indicator for JPM", fontsize=20)
    ax.legend(fontsize=20)

    ax2.plot(price["Date"], price[symbol])
    ax2.set_ylabel("Normalized Price", fontsize=20)

    unique_months = sso["Date"].dt.strftime("%b %Y").unique()
    x_ticks = [
        sso[sso["Date"].dt.strftime("%b %Y") == month]["Date"].iloc[0]
        for month in unique_months
    ]
    x_labels = [month for month in unique_months]
    plt.xticks(x_ticks, x_labels, rotation=45, fontsize=20)
    plt.grid(True)
    plt.xlabel("Date", fontsize=20)
    plt.subplots_adjust(top=0.95, bottom=0.20)
    plt.savefig("Figure6.png")


def author():
    return "karcot3"
