import datetime as dt
import warnings

import marketsimcode
import numpy as np
import pandas as pd
import util
from QRLTrader.StrategyLearner import StrategyLearner as QRLTrader
from QTrader.StrategyLearner import StrategyLearner as QTrader

warnings.filterwarnings("ignore")
symbol = "SPY"
start_val = 100000
# np.random.seed(42)

tsd = dt.date(2011, 1, 1)
ted = dt.date(2014, 12, 31)

sd = dt.date(2015, 1, 1)
ed = dt.date(2016, 12, 31)
l = []
for c in [QTrader, QRLTrader]:

    trader = c()

    trader.add_evidence(symbol, tsd, ted)

    trades = trader.testPolicy(sd, ed, sv=start_val)

    port_vals = marketsimcode.compute_portvals(trades, symbol, start_val, sd, ed)
    print(type(trader))
    # cr = port_vals.iloc[-1, 0] / port_vals.iloc[0, 0] - 1
    # print(cr)
    print(port_vals.iloc[-1, 0])
    print(port_vals.iloc[0, 0])
    l.append(port_vals)
    util.get_stats(port_vals)

df = util.get_data(symbol, sd, ed)
# RL,QRL,Benchmark
df["RL"] = l[0]
df["QRL"] = l[1]

first_day = df.index[0]
trades_benchmark = pd.DataFrame({"Date": [first_day], "Order": [200]})
trades_benchmark.set_index("Date", inplace=True)

port_vals = marketsimcode.compute_portvals(trades_benchmark, symbol, start_val, sd, ed)
print("Benchmark")
print(port_vals.iloc[-1, 0])
print(port_vals.iloc[0, 0])
print("Benchmark")
util.get_stats(port_vals)
df["Benchmark"] = port_vals
util.plotReturns(df, symbol)

print("Done")
