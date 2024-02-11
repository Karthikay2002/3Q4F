import datetime as dt
import warnings

import util
from marketsimcode import compute_portvals
from StrategyLearner import StrategyLearner

# Ignore all warnings
warnings.filterwarnings("ignore")
symbol = "JPM"
learner = StrategyLearner()

learner.add_evidence(symbol=symbol)

sd = dt.date(2022, 1, 1)
ed = dt.date(2022, 12, 31)
trades = learner.testPolicy(symbol=symbol)

portvals = compute_portvals(
    trades,
    symbol,
    start_val=100000,
    start_date=sd,
    end_date=ed,
)
print(len(trades))
cr = portvals.iloc[-1] / portvals.iloc[0] - 1
print("Final Value")
print(cr)
price = util.get_data(symbol, sd, ed)
price = price[[symbol]]
print("Default Value")
cr = price.iloc[-1] / price.iloc[0] - 1
print(cr)
