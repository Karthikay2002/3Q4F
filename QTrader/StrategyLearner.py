import datetime as dt
import random

import numpy as np
import pandas as pd
import util as ut
from marketsimcode import compute_portvals
from QTrader.indicators import (
    getBollingerBandsIndicator,
    getEMAIndicator,
    getStochasticOscillatorIndicator,
)
from QTrader.QLearner import QLearner


class StrategyLearner(object):
    """
    A strategy learner that can learn a trading policy using the same indicators used in ManualStrategy.

    :param verbose: If “verbose” is True, your code can print out information for debugging.
        If verbose = False your code should not generate ANY output.
    :type verbose: bool
    :param impact: The market impact of each transaction, defaults to 0.0
    :type impact: float
    :param commission: The commission amount charged, defaults to 0.0
    :type commission: float
    """

    # constructor
    def __init__(self, verbose=False, impact=0.0, commission=0.0):
        """
        Constructor method
        """
        self.verbose = verbose
        self.impact = impact
        self.commission = commission
        self.bblookback = 20
        self.emalookback = 14
        self.ssolookback = 20
        self.window = 10
        self.ybuy = 0.2 + self.impact
        self.ysell = -0.2 - self.impact
        self.state_map = {}
        u = 0
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                for k in [-1, 0, 1]:
                    for l in [-1, 0, 1]:
                        self.state_map[(i, j, k, l)] = u
                        u += 1
        self.learner = QLearner(
            num_states=len(self.state_map),
            num_actions=3,
            alpha=0.5,
            gamma=0.8,
            rar=0.5,
            radr=0.99,
            dyna=0,
        )

    def return_state(self, bbI, emaI, ssoI, stock_state):
        return self.state_map[(bbI, emaI, ssoI, stock_state)]

    def get_reward(self, a, y, cs):
        r = 0.0
        if a == 1 and y == 1:
            if cs == -1:
                r = 3.0
            elif cs == 0:
                r = 2.0
            elif cs == 1:
                r = 1.0
        elif a == -1 and y == -1:
            if cs == 1:
                r = 3.0
            elif cs == 0:
                r = 2.0
            elif cs == -1:
                r = 1.0
        elif a == 0:
            if y != 0:
                r = -1.0
        elif a == 1 and y == -1:
            if cs == 1:
                r = -3.0
            elif cs == 0:
                r = -2.0
            else:
                r = -1.0
        elif a == -1 and y == 1:
            if cs == -1:
                r = -3.0
            elif cs == 0:
                r = -2.0
            elif cs == 1:
                r = -1.0
        return r

    # this method should create a QLearner, and train it for trading
    def add_evidence(
        self,
        symbol="JPM",
        sd=dt.datetime(2018, 1, 1),
        ed=dt.datetime(2021, 12, 31),
        sv=100000,
    ):
        """
        Trains your strategy learner over a given time frame.

        :param symbol: The stock symbol to train on
        :type symbol: str
        :param sd: A datetime object that represents the start date, defaults to 1/1/3008
        :type sd: datetime
        :param ed: A datetime object that represents the end date, defaults to 1/1/3009
        :type ed: datetime
        :param sv: The starting value of the portfolio
        :type sv: int
        """

        # add your code to do learning here

        # example usage of the old backward compatible util function
        # syms = [symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(symbol, sd, ed)  # automatically adds SPY
        # print(prices_all.columns)
        prices = prices_all[[symbol]]  # only portfolio symbols
        parray = prices[symbol].values
        y = parray.copy()
        self.symbol = symbol
        for i in range(len(y) - self.window):
            y[i] = 0
            # print(str(parray[i+self.window]/(parray[i] + self.commission) - 1.0) + ' ' + str(self.ybuy))
            # print(str(parray[i+self.window]/(parray[i]-self.commission) - 1.0) + ' ' + str(self.ysell))
            # print(str(parray[i+self.window]) + ' ' + str(parray[i]))
            if (
                parray[i + self.window] / (parray[i] + self.commission / 300.0) - 1.0
                > self.ybuy
            ):
                y[i] = 1
            elif (parray[i + self.window] + self.commission / 300.0) / parray[
                i
            ] - 1.0 < self.ysell:
                y[i] = -1
        y = y[1 : len(parray) - self.window]
        # print(y)
        bbI = getBollingerBandsIndicator(sd, ed, symbol, lookback=self.bblookback)[
            : -(self.window + 1)
        ]
        emaI = getEMAIndicator(sd, ed, symbol, lookback=self.emalookback)[
            : -(self.window + 1)
        ]
        ssoI = getStochasticOscillatorIndicator(
            sd, ed, symbol, lookback=self.ssolookback
        )[: -(self.window + 1)]
        for _ in range(100):
            s = self.return_state(
                bbI[symbol][0], emaI[symbol][0], ssoI[symbol][0], 0
            )  # current_date)
            a = self.learner.querysetstate(s)
            r = self.get_reward(a, y[0], 0)
            cs = 0
            if a == 1:
                cs = 1
            elif a == -1:
                cs = -1
            for i in range(1, len(y)):
                new_state = self.return_state(
                    bbI[symbol][i], emaI[symbol][i], ssoI[symbol][i], cs
                )
                a = self.learner.query(new_state, r) - 1
                r = self.get_reward(a, y[i], cs)
                if cs == 1 and a == -1:
                    cs = 0
                elif cs == -1 and a == 1:
                    cs = 0
                elif cs == 0 and a == 1:
                    # print("Yo " + str(r) + " " + str(y[i]))
                    cs = 1
                elif cs == 0 and a == -1:
                    # print("Ssup")
                    cs = -1
            trades = self.testPolicy(sd, ed)
            portvals = compute_portvals(
                trades,
                symbol,
                start_val=100000,
                start_date=sd,
                end_date=ed,
            )
            # print(cr)
            # print(len(trades))
            cr = portvals.iloc[-1] / portvals.iloc[0] - 1
            # print(cr)

    # this method should use the existing policy and test it against new data
    def testPolicy(
        self,
        sd=dt.datetime(2022, 1, 1),
        ed=dt.datetime(2022, 12, 31),
        sv=100000,
    ):
        """
        Tests your learner using data outside of the training data

        :param symbol: The stock symbol that you trained on on
        :type symbol: str
        :param sd: A datetime object that represents the start date, defaults to 1/1/3008
        :type sd: datetime
        :param ed: A datetime object that represents the end date, defaults to 1/1/3009
        :type ed: datetime
        :param sv: The starting value of the portfolio
        :type sv: int
        :return: A DataFrame with values representing trades for each day. Legal values are +300.0 indicating
            a BUY of 1000 shares, -300.0 indicating a SELL of 1000 shares, and 0.0 indicating NOTHING.
            Values of +3000 and -3000 for trades are also legal when switching from long to short or short to
            long so long as net holdings are constrained to -1000, 0, and 1000.
        :rtype: pandas.DataFrame
        """

        # here we build a fake set of trades
        # your code should return the same sort of data
        dates = pd.date_range(sd, ed)
        bbI = getBollingerBandsIndicator(sd, ed, self.symbol, lookback=self.bblookback)
        emaI = getEMAIndicator(sd, ed, self.symbol, lookback=self.emalookback)
        ssoI = getStochasticOscillatorIndicator(
            sd, ed, self.symbol, lookback=self.ssolookback
        )
        trades_list = []
        dates = bbI.index
        flag = 0
        for i in range(len(dates) - 1):
            y_test = 0
            s = self.return_state(
                bbI[self.symbol][i], emaI[self.symbol][i], ssoI[self.symbol][i], flag
            )
            y_test = self.learner.querysetstate(s) - 1
            if flag == 0:
                if y_test == 1:
                    trades_list.append([dates[i + 1], 300.0])
                    flag = 1
                elif y_test == -1:
                    trades_list.append([dates[i + 1], -300.0])
                    flag = -1
            elif flag == 1:
                if y_test == -1:
                    trades_list.append([dates[i + 1], -300.0])
                    flag = 0
            elif flag == -1:
                if y_test == 1:
                    trades_list.append([dates[i + 1], 300.0])
                    flag = 0
        trades = pd.DataFrame(trades_list, columns=["Date", "Order"])
        trades.set_index("Date", inplace=True)
        return trades


if __name__ == "__main__":
    print("One does not simply think up a strategy")
