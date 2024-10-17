import pandas as pd
import numpy as np
import logging
from misc.misc import *
from core.portfolio_generator import PortfolioGenerator
from core.file_adapter import FileAdapter
from core import logging_config

def return_cleaning(returns):
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
    returns_clean = returns_clean.iloc[1:]
    returns_clean = returns_clean.fillna(0)
    return returns_clean

def build_weights(max_sharpe, min_var, custom, ticks):
    """Function builds dictionary containing all relevant
    portfolio weights. This inlcudes weights for maximum
    sharpe ratio, minimum variance, custom defined weights
    and equal weights. Function also considers error handling
    for the custom defined weights

    :param max_sharpe: Maximum sharpe ratio weights
    :type max_sharpe: Dictionary
    :param min_var: Minimum variance weights
    :type min_var: Dictionary
    :param custom: Custom defined weights
    :type custom: Dictionary
    :param ticks: Harmonized ticker symbols
    :type ticks: List
    :return: All relevant portfolio weights for each ticker
    :rtype: Dictionary
    """
    custom_weights = custom.copy()
    custom_weight_sum = 0
    for tick, custom_weight in custom.items():
        if tick not in ticks:
            logging.warning(f"Ticker {tick} not in ticker list for historical returns")
            custom_weights.pop(tick)
            logging.info(f"Ticker {tick} was removed from custom weights")
        else:
            custom_weight_sum += custom_weight
    custom_weighs_ticks = custom_weights.keys()
    if custom_weight_sum != 1 or len(ticks) != len(custom_weighs_ticks):
        logging.warning("Sum of custom weights are not equal to 1 or length of custom weights not match with portfolio constituents")

    equal_weights = {}
    equal_weight = 1 / len(ticks)
    equal_weight = np.round(equal_weight, 3)
    for tick in ticks:
        equal_weights[tick] = equal_weight

    weights = {}
    weights[CONST_KEYS["MAX_SHARPE"]] = max_sharpe
    weights[CONST_KEYS["MIN_VAR"]] = min_var
    weights[CONST_KEYS["CUSTOM"]] = custom_weights
    weights[CONST_KEYS["EQUAL"]] = equal_weights
    return weights

def build_historical_portfolios(returns, weights):
    for key, weight_dict in weights.items():
        port_rets = PortfolioGenerator(returns).get_returns(weights)
        print("break")
    

if __name__ == "__main__":
    CONST_COLS = read_json("constant.json")["columns"]
    CONST_KEYS = read_json("constant.json")["keys"]
    PARAMETER = read_json("parameter.json")
    raw_stock_returns = FileAdapter().load_stock_returns()
    raw_benchmark_returns = FileAdapter().load_benchmark_returns()
    raw_stock_returns, tickers = harmonize_tickers(raw_stock_returns)
    stock_returns = return_cleaning(raw_stock_returns)
    benchmark_returns = return_cleaning(raw_benchmark_returns)
    max_sharpe_weights = PortfolioGenerator(stock_returns).get_max_sharpe_weights()
    min_var_weights = PortfolioGenerator(stock_returns).get_min_var_weights()
    custom_weights = PARAMETER["custom_weights"]
    weight_dict = build_weights(max_sharpe_weights, min_var_weights, custom_weights, tickers)
    hist_port_rets = build_historical_portfolios(stock_returns, weight_dict)
    
    
