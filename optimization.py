""""""  		  	   		  		 		  		  		    	 		 		   		 		  
"""MC1-P2: Optimize a portfolio.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		  		 		  		  		    	 		 		   		 		  
Atlanta, Georgia 30332  		  	   		  		 		  		  		    	 		 		   		 		  
All Rights Reserved  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
Template code for CS 4646/7646  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		  		 		  		  		    	 		 		   		 		  
works, including solutions to the projects assigned in this course. Students  		  	   		  		 		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		  	   		  		 		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		  	   		  		 		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		  	   		  		 		  		  		    	 		 		   		 		  
or edited.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
We do grant permission to share solutions privately with non-students such  		  	   		  		 		  		  		    	 		 		   		 		  
as potential employers. However, sharing with other current or future  		  	   		  		 		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		  		 		  		  		    	 		 		   		 		  
GT honor code violation.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
-----do not edit anything above this line---  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
Student Name: Kaushik Arcot(replace with your name)  		  	   		  		 		  		  		    	 		 		   		 		  
GT User ID: karcot3 (replace with your User ID)  		  	   		  		 		  		  		    	 		 		   		 		  
GT ID: 903944559 (replace with your GT ID)  		  	   		  		 		  		  		    	 		 		   		 		  
"""  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
import datetime as dt  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
import numpy as np  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
import matplotlib.pyplot as plt  		  	   		  		 		  		  		    	 		 		   		 		  
import pandas as pd  		  	   		  		 		  		  		    	 		 		   		 		  
from util import get_data, plot_data
from scipy.optimize import minimize


def negative_sharpe_ratio(allocs, prices, syms):
    """
    :param prices: Data of price at close of day
    :param allocs: allocations of each stock in the portfolio
    :return: Negative sharpe ratio for the portfolio
    """
    daily_returns = (prices/prices.shift(1)) - 1.0
    daily_returns.iloc[0,:] = 0.0
    daily_returns = daily_returns[1:]
    k = 252.0
    var = 0.0
    m = 0.0
    portfolio_daily_returns = np.zeros(len(daily_returns))
    for i,col in enumerate(syms):
        var = var + (allocs[i]**2)*(daily_returns[col].std()**2)
        m = m + allocs[i]*daily_returns[col].mean()
        portfolio_daily_returns += allocs[i]*daily_returns[col].values
    for i in range(len(syms)):
        col1 = syms[i]
        for j in range(i+1,len(syms)):
            col2 = syms[j]
            var = var + 2*allocs[i]*allocs[j]*daily_returns[col1].cov(daily_returns[col2])
    std_dev = np.sqrt(var)
    # std_dev = portfolio_daily_returns.std()
    sharpe_ratio = (np.sqrt(k) * m )/ std_dev
    # print(sharpe_ratio)
    return -1*sharpe_ratio

  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
# This is the function that will be tested by the autograder  		  	   		  		 		  		  		    	 		 		   		 		  
# The student must update this code to properly implement the functionality  		  	   		  		 		  		  		    	 		 		   		 		  
def optimize_portfolio(  		  	   		  		 		  		  		    	 		 		   		 		  
    sd=dt.datetime(2008, 6, 1),
    ed=dt.datetime(2009, 6, 1),
    syms= ['IBM', 'X', 'GLD', 'JPM'],
    gen_plot=False,  		  	   		  		 		  		  		    	 		 		   		 		  
):  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    This function should find the optimal allocations for a given set of stocks. You should optimize for maximum Sharpe  		  	   		  		 		  		  		    	 		 		   		 		  
    Ratio. The function should accept as input a list of symbols as well as start and end dates and return a list of  		  	   		  		 		  		  		    	 		 		   		 		  
    floats (as a one-dimensional numpy array) that represents the allocations to each of the equities. You can take  		  	   		  		 		  		  		    	 		 		   		 		  
    advantage of routines developed in the optional assess portfolio project to compute daily portfolio value and  		  	   		  		 		  		  		    	 		 		   		 		  
    statistics.  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		  		 		  		  		    	 		 		   		 		  
    :type sd: datetime  		  	   		  		 		  		  		    	 		 		   		 		  
    :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		  		 		  		  		    	 		 		   		 		  
    :type ed: datetime  		  	   		  		 		  		  		    	 		 		   		 		  
    :param syms: A list of symbols that make up the portfolio (note that your code should support any  		  	   		  		 		  		  		    	 		 		   		 		  
        symbol in the data directory)  		  	   		  		 		  		  		    	 		 		   		 		  
    :type syms: list  		  	   		  		 		  		  		    	 		 		   		 		  
    :param gen_plot: If True, optionally create a plot named plot.png. The autograder will always call your  		  	   		  		 		  		  		    	 		 		   		 		  
        code with gen_plot = False.  		  	   		  		 		  		  		    	 		 		   		 		  
    :type gen_plot: bool  		  	   		  		 		  		  		    	 		 		   		 		  
    :return: A tuple containing the portfolio allocations, cumulative return, average daily returns,  		  	   		  		 		  		  		    	 		 		   		 		  
        standard deviation of daily returns, and Sharpe ratio  		  	   		  		 		  		  		    	 		 		   		 		  
    :rtype: tuple  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    # Read in adjusted closing prices for given symbols, date range  		  	   		  		 		  		  		    	 		 		   		 		  
    dates = pd.date_range(sd, ed)  		  	   		  		 		  		  		    	 		 		   		 		  
    prices_all = get_data(syms, dates)  # automatically adds SPY  		  	   		  		 		  		  		    	 		 		   		 		  
    prices = prices_all[syms]  # only portfolio symbols  		  	   		  		 		  		  		    	 		 		   		 		  
    prices_SPY = prices_all["SPY"]  # only SPY, for comparison later  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    # find the allocations for the optimal portfolio  		  	   		  		 		  		  		    	 		 		   		 		  
    # note that the values here ARE NOT meant to be correct for a test case
    n = len(syms)
    init_allocs = np.full((n), 1.0/n) # add code here to find the allocations
    bnds = tuple(((0.0,1.0) for _ in range(n)))
    cons = ({'type': 'eq', 'fun': lambda inputs: 1.0 - np.sum(inputs)})
    res = minimize(negative_sharpe_ratio,x0=init_allocs, bounds=bnds, args=(prices,syms),
                   constraints=cons, method = 'SLSQP')
    allocs = res.x
    daily_returns = (prices / prices.shift(1)) - 1.0
    daily_returns.iloc[0, :] = 0.0
    daily_returns = daily_returns[1:]
    portfolio_daily_returns = np.zeros(len(daily_returns))

    portfolio_daily_prices = np.zeros(len(prices))
    for i, col in enumerate(syms):
        portfolio_daily_prices += allocs[i] * prices[col].values
        portfolio_daily_returns += allocs[i] * daily_returns[col].values
    # Compare daily portfolio value with SPY using a normalized plot
    port_val = prices.copy()
    port_val['Portfolio'] = portfolio_daily_prices
    port_val = port_val[['Portfolio']]
    if gen_plot:  		  	   		  		 		  		  		    	 		 		   		 		  
        # add code to plot here  		  	   		  		 		  		  		    	 		 		   		 		  
        df_temp = pd.concat(  		  	   		  		 		  		  		    	 		 		   		 		  
            [port_val, prices_SPY], axis=1
        )
        df_temp = df_temp/df_temp.iloc[0]
        df_temp.index = pd.to_datetime(df_temp.index).strftime('%b%Y')
        ax = df_temp.plot(title = 'Daily Portfolio Value Compared with SPY')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        plt.savefig('images/Figure1.png')
    adr = np.mean(portfolio_daily_returns)
    sddr = np.std(portfolio_daily_returns)
    sr = -1*res.fun
    cr = port_val['Portfolio'].iloc[-1]/port_val['Portfolio'].iloc[0] - 1
    # AXP = 0.78
    # HPQ = 0.22
    # print(allocs)
    # print(syms)
    return allocs, cr, adr, sddr, sr  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
def test_code():  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
    This function WILL NOT be called by the auto grader.  		  	   		  		 		  		  		    	 		 		   		 		  
    """  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    start_date = dt.datetime(2008, 6, 1)
    end_date = dt.datetime(2009, 6, 1)
    symbols= ['IBM', 'X', 'GLD', 'JPM']
  		  	   		  		 		  		  		    	 		 		   		 		  
    # Assess the portfolio  		  	   		  		 		  		  		    	 		 		   		 		  
    allocations, cr, adr, sddr, sr = optimize_portfolio(  		  	   		  		 		  		  		    	 		 		   		 		  
        sd=start_date, ed=end_date, syms=symbols, gen_plot=True
    )  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
    # Print statistics
    # print(allocations)
    print(f"Start Date: {start_date}")
    print(f"End Date: {end_date}")
    print(f"Symbols: {symbols}")
    print(f"Allocations:{allocations}")
    print(f"Sharpe Ratio: {sr}")
    print(f"Volatility (stdev of daily returns): {sddr}")
    print(f"Average Daily Return: {adr}")
    print(f"Cumulative Return: {cr}")
    # 0.46, 0.37, 0.0, 0.17
  		  	   		  		 		  		  		    	 		 		   		 		  
  		  	   		  		 		  		  		    	 		 		   		 		  
if __name__ == "__main__":  		  	   		  		 		  		  		    	 		 		   		 		  
    # This code WILL NOT be called by the auto grader  		  	   		  		 		  		  		    	 		 		   		 		  
    # Do not assume that it will be called  		  	   		  		 		  		  		    	 		 		   		 		  
    test_code()  		  	   		  		 		  		  		    	 		 		   		 		  
