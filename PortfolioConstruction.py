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
    """Function constructs the actual portfoliow weights
    based on the optimal weights and the last actual
    closing quote for each tick and each portfolio type. 
    Based on the actual portfolio weights and the last 
    observed closing quote, the function calculates the 
    actual long position as number of shares to buy and 
    the investment amount for each stock. Both measures
    are saved in a list, whereby the first entry displays
    the number of shares to buy and the second entry the
    investment amount.

    :param opt_weights: Optimal portfolio weights
    :type opt_weights: Dictionary
    :return: Actual portfolio weights, Actual long position
    :rtype: Dictionary, Dictionary
    """
    invest = PARAMETER["investment"]
    actual_weights = {}
    actual_long_pos = {}
    for port_type, weights in opt_weights.items():
        weight_dict = {}
        long_pos_dict = {}
        for tick, weight in weights.items():
            long_pos_list = []
            # actual_quotes = FinanceAdapter(tick).get_last_quote()
            # actual_quotes = rename_yfcolumns(data=actual_quotes)
            # actual_quote = actual_quotes[PARAMETER["quote_id"]]
            # actual_quote = actual_quote.iloc[0]
            actual_quote = 100
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

def get_cumulative_returns(returns):
    """Function calculates the cumulative and
    exponential portfolio returns. The returns 
    are multiplied by factor 1000 for better
    visualization.

    :param returns: Daily portfolio returns
    :type returns: Dataframe
    :return: Cumulative portfolio returns
    :rtype: Dataframe
    """
    cum_returns = pd.DataFrame()
    for tick, rets in returns.items():
        cum_rets = rets.cumsum().apply(np.exp)
        cum_rets = cum_rets * 1000
        cum_returns[tick] = cum_rets
    return cum_returns

def calculate_performance(port_rets, bench_rets):
    """Function calculates annualized performance measures
    for given daily portfolio returns and benchmark returns

    :param port_rets: Daily portfolio returns
    :type port_rets: Series
    :param bench_rets: Daily benchmark returns
    :type bench_rets: Series
    :return: Annualized performance measures
    :rtype: Dictionary
    """
    ann_mean_ret = port_rets.mean() * 252
    ann_vola = np.sqrt(port_rets.std() * 252)
    sharpe_ratio = ann_mean_ret / ann_vola
    bench_corr = port_rets.corr(bench_rets)
    performance_dict = {}
    performance_dict[CONST_COLS["sharpe_ratio"]] = sharpe_ratio
    performance_dict[CONST_COLS["ann_mean_ret"]] = ann_mean_ret
    performance_dict[CONST_COLS["ann_vola"]] = ann_vola
    performance_dict[CONST_COLS["bench_corr"]] = bench_corr
    return performance_dict

def build_portfolio_performance(port_rets, bench_rets):
    """Function constructs performance measures for
    all given portfolio types and the respective portfolio
    returns and benchmark returns

    :param port_rets: Daily portfolio type returns
    :type port_rets: Dataframe
    :param bench_rets: Daily benchmark returns
    :type bench_rets: Dataframe
    :return: Performance measures
    :rtype: Dataframe
    """
    results = pd.DataFrame()
    bench_rets = bench_rets.squeeze()
    total_rets = pd.concat(
        [port_rets, bench_rets], 
        axis=1
    )
    for port_type, rets in total_rets.items():
        perf_dict = calculate_performance(
            port_rets=rets,
            bench_rets=bench_rets
        )
        for measure, value in perf_dict.items():
            results.loc[port_type, measure] = value
    results = results.round(3)
    return results

def build_long_position(opt_weights, act_weights, long_pos, ticks):
    """Function constructs long position values for given
    portfolio types and stock tickers

    :param opt_weights: Optimal portfolio weights for respective portfolio types
    :type opt_weights: Dictionary
    :param act_weights: Actual portfolio weights for respective portfolio types
    :type act_weights: Dictionary
    :param long_pos: Actual long position values for respective portfolio types
    :type long_pos: Dictionary
    :param ticks: Stock tickers
    :type ticks: List
    :return: Long position values
    :rtype: Dictionary
    """
    opt_dict_keys = list(opt_weights.keys())
    act_dict_keys = list(act_weights.keys())
    long_pos_keys = list(long_pos.keys())
    common_keys = get_list_intersection(
        opt_dict_keys,
        act_dict_keys,
        long_pos_keys
    )
    result_dict = {}
    for type in common_keys:
        long_pos_results = pd.DataFrame(
            index=ticks
        )
        for tick in ticks:
            opt_weight = opt_weights[type].get(tick)
            act_weight = act_weights[type].get(tick)
            n_shares = long_pos[type].get(tick)[0]
            invest = long_pos[type].get(tick)[1]
            long_pos_results.loc[tick, CONST_COLS["opt_weight"]] = opt_weight
            long_pos_results.loc[tick, CONST_COLS["act_weight"]] = act_weight
            long_pos_results.loc[tick, CONST_COLS["long_pos"]] = n_shares
            long_pos_results.loc[tick, CONST_COLS["amount"]] = invest
        result_dict[type] = long_pos_results
    return result_dict


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
    historical_portfolio_returns = build_historical_portfolios(stock_returns, actual_weights)
    future_portfolio_returns = build_future_portfolios(tickers, actual_weights)
    cumulative_benchmark_returns = get_cumulative_returns(benchmark_returns)
    cumulative_historical_returns = get_cumulative_returns(historical_portfolio_returns)
    cumulative_future_returns = get_cumulative_returns(future_portfolio_returns)
    portfolio_performance = build_portfolio_performance(historical_portfolio_returns, benchmark_returns)
    long_position = build_long_position(optimal_weights, actual_weights, actual_long_position, tickers)
    
    
