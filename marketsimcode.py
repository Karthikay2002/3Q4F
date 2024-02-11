import pandas as pd
from util import get_data


def author():
    return "karcot3"


def find_current_value(current_portfolio, prices, cash, date):
    val = 0
    for k in current_portfolio:
        val = val + prices[k][date] * current_portfolio[k]
    val = val + cash
    return val


def compute_portvals(
    orders, symbol, start_val, start_date, end_date, impact=0, commission=0
):
    """
    Computes the portfolio values.

    :param orders_file: Path of the order file or the file object
    :type orders_file: str or file object
    :param start_val: The starting value of the portfolio
    :type start_val: int
    :param commission: The fixed amount in dollars charged for each transaction (both entry and exit)
    :type commission: float
    :param impact: The amount the price moves against the trader compared to the historical data at each transaction
    :type impact: float
    :return: the result (portvals) as a single-column dataframe, containing the value of the portfolio for each trading day in the first column from start_date to end_date, inclusive.
    :rtype: pandas.DataFrame
    """
    symbols = [symbol]
    dates = pd.date_range(start_date, end_date)
    prices = get_data(symbol, start_date, end_date)
    dates = prices.index
    prices = prices[symbols]
    cash = start_val
    port_vals = []
    current_portfolio = {}
    for date in dates:
        if date in orders.index:
            curr_orders = orders[orders.index == date]
            for _, row in curr_orders.iterrows():
                num = row["Order"]
                curr_impact = impact
                if num < 0:
                    curr_impact = impact * -1
                if symbol not in current_portfolio:
                    current_portfolio[symbol] = num
                else:
                    current_portfolio[symbol] = current_portfolio[symbol] + num
                cash = (
                    cash - num * prices[symbol][date] * (1 + curr_impact) - commission
                )
            # print(cash)
        port_vals.append(find_current_value(current_portfolio, prices, cash, date))
    portvals = pd.DataFrame(index=dates)
    portvals["Value"] = port_vals
    return portvals
