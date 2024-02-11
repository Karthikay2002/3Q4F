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
