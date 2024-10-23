import pandas as pd
import numpy as np
import logging
import math
from misc.misc import *
from core.portfolio_generator import PortfolioGenerator
from core.file_adapter import FileAdapter
from core.finance_adapter import FinanceAdapter
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

def build_optimal_weights(max_sharpe, min_var, ticks):
    """Function builds dictionary containing all relevant
    portfolio weights. This inlcudes weights for maximum
    sharpe ratio, minimum variance, custom defined weights
    and equal weights. Function also considers error handling
    for the custom defined weights

    :param max_sharpe: Maximum sharpe ratio weights
    :type max_sharpe: Dictionary
    :param min_var: Minimum variance weights
    :type min_var: Dictionary
    :param ticks: Harmonized ticker symbols
    :type ticks: List
    :return: All relevant portfolio weights for each ticker
    :rtype: Dictionary
    """
    custom = PARAMETER["custom_weights"]
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

def build_historical_portfolios(hist_returns, weights):
    """Function calculates and concats daily historical
    portfolio returns for given weights

    :param hist_returns: Historical stock returns
    :type hist_returns: Dataframe
    :param weights: Weights for portfolio calculation
    :type weights: Dictionary
    :return: Historical portfolio returns
    :rtype: Dataframe
    """
    hist_ports_list = []
    for key, weight_dict in weights.items():
        port_rets = PortfolioGenerator(hist_returns).get_returns(weight_dict)
        port_rets.name = key
        hist_ports_list.append(port_rets)
    hist_ports = pd.concat(hist_ports_list, axis=1)
    return hist_ports
    
def build_future_portfolios(tickers, weights):
    """Function loads trained models for given tickers and 
    defined model type in parameter file and calculates and
    concats daily future portfolio returns for given weights

    :param tickers: Tickers for future return calculation
    :type tickers: List
    :param weights: Weights for portfolio calculation
    :type weights: Dictionary
    :return: Future portfolio returns 
    :rtype: Dataframe
    """
    return_list = []
    for tick in tickers:
        model_id = get_latest_modelid(
            tick=tick, 
            model_type=PARAMETER["model"]
        )
        model = FileAdapter().load_model(
            ticker=tick,
            model_id=model_id
        )
        quote_prediction = model.predict()
        returns = calculate_returns(quotes=quote_prediction)
        returns.name = tick
        return_list.append(returns)

    future_ports_list = []
    future_returns = pd.concat(return_list, axis=1)
    for key, weight_dict in weights.items():
        port_rets = PortfolioGenerator(future_returns).get_returns(weight_dict)
        port_rets.name = key
        future_ports_list.append(port_rets)
    future_ports = pd.concat(future_ports_list, axis=1)
    return future_ports

def build_actual_values(opt_weights):
    invest = PARAMETER["investment"]
    actual_weights = {}
    actual_long_pos = {}
    for port_type, weights in opt_weights.items():
        weight_dict = {}
        long_pos_dict = {}
        for tick, weight in weights.items():
            long_pos_list = []
            actual_quotes = FinanceAdapter(tick).get_last_quote()
            actual_quotes = rename_yfcolumns(data=actual_quotes)
            actual_quote = actual_quotes[PARAMETER["quote_id"]]
            actual_quote = actual_quote.iloc[0]
            raw_amount = invest * weight
            n_shares = raw_amount / actual_quote
            n_shares_adj = int(math.floor(n_shares))
            weight_adj = actual_quote * n_shares_adj
            weight_adj = weight_adj / invest
            weight_adj = np.round(weight_adj, 3)
            tick_invest = n_shares_adj * actual_quote
            tick_invest = np.round(tick_invest, 3)
            long_pos_list.append(n_shares_adj)
            long_pos_list.append(tick_invest)
            weight_dict[tick] = weight_adj
            long_pos_dict[tick] = long_pos_list
        actual_weights[port_type]  = weight_dict
        actual_long_pos[port_type] = long_pos_dict
    return actual_weights, actual_long_pos

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
    optimal_weights = build_optimal_weights(max_sharpe_weights, min_var_weights, tickers)
    actual_weights, actual_long_position = build_actual_values(optimal_weights)
    hist_port_rets = build_historical_portfolios(stock_returns, optimal_weights)
    future_port_rets = build_future_portfolios(tickers, optimal_weights)
    print("break")
    
    
