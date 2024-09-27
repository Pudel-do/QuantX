import pandas as pd
import numpy as np
from misc.misc import *
from core.portfolio_generator import PortfolioGenerator
from core.file_adapter import FileAdapter

def data_cleaning(returns):
    """Function preprocesses return data
    for clean data set as base for portfolio
    construction. Preprocessing includes
    portfolio period filtern and fill nan values

    :param returns: Stock returns
    :type returns: Dataframe
    :return: Clean returns data
    :rtype: Dataframe
    """
    period_mask = returns.index >= PARAMETER["portfolio_start"]
    returns_clean = returns[period_mask]
    returns_clean = returns_clean.fillna(method="ffill")
    returns_clean = returns_clean.dropna()
    return returns_clean

if __name__ == "__main__":
    PARAMETER = read_json("parameter.json")
    returns = FileAdapter().load_returns()
    returns, tickers = harmonize_tickers(returns)
    returns_clean = data_cleaning(returns)
