import pandas as pd
import numpy as np
import logging
from misc.misc import *
from core.portfolio_generator import PortfolioGenerator
from core.file_adapter import FileAdapter
from core import logging_config

def data_cleaning(returns):
    """Function preprocesses return data
    for clean data set as base for portfolio
    construction. Preprocessing includes
    portfolio period filtering and fill nan
    values with zero

    :param returns: Stock returns
    :type returns: Dataframe
    :return: Clean returns data
    :rtype: Dataframe
    """
    period_mask = returns.index >= PARAMETER["portfolio_start"]
    returns_clean = returns[period_mask]
    returns_clean = returns_clean.iloc[1:, :]
    returns_clean = returns_clean.fillna(0)
    return returns_clean

def validate_custom_weights(weights, tickers):
    weights_clean = weights.copy()
    weight_sum = 0
    for tick, weight in weights.items():
        weight_sum =+ weight
        if tick not in tickers:
            logging.warning(f"Ticker {tick} not in historical returns")
            weights_clean.pop(tick)
        else:
            pass
    #ToDo: Check if sum of weights equal zero or lenght of both ticker symbols are the same
    #ToDo: If not, create work around for stabil code
    if np.round(weight_sum) != 1:
        pass
    return weights_clean

if __name__ == "__main__":
    PARAMETER = read_json("parameter.json")
    raw_returns = FileAdapter().load_returns()
    raw_returns, tickers = harmonize_tickers(raw_returns)
    hist_returns = data_cleaning(raw_returns)
    max_sharpe_weights = PortfolioGenerator(returns=hist_returns).get_max_sharpe_weights()
    min_var_weights = PortfolioGenerator(returns=hist_returns).get_min_var_weights()
    custom_weights = PARAMETER["custom_weights"]
    custom_weights = validate_custom_weights(weights=custom_weights, tickers=tickers)
    
