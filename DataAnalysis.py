import pandas as pd
import numpy as np
from itertools import product
from core.file_adapter import FileAdapter
from core.dashboard_adapter import AnalysisDashboard
from misc.misc import *

def get_moving_averages(quote_data, train):
    """Function calculates moving averages as well as
    historical trading strategy returns and determines
    the strategy outperformance by comparing strategy
    returns with market returns for the same stock

    :param quote_data: Historical stock quote data
    :type quote_data: Series
    :param train: Parameter to control for training and testing 
    samples when calculating the trading strategy performance and
    the underlying moving average parameters
    :type train: Boolean
    :return: Stock quotes extended by moving average quotes and market
    and strategy returns as well as optimal moving average parameters
    :rtype: Dictionary, Series
    """
    moving_average_dict = {}
    optimal_ma_value_list = []
    for ticker, quotes in quote_data.items():
        quotes.name = QUOTE_COL
        if train:
            train_set, test_set = ts_train_test_split(
                ts=quotes,
                train_ratio=0.75
            )
            opt_values = ma_optimization(quotes=train_set)
            sma1 = int(opt_values.loc["SMA1"])
            sma2 = int(opt_values.loc["SMA2"])
            backtest = vectorized_backtesting(
                quotes=test_set, 
                SMA1=sma1, 
                SMA2=sma2
            )
        else:
            opt_values = ma_optimization(quotes=quotes)
            sma1 = int(opt_values.loc["SMA1"])
            sma2 = int(opt_values.loc["SMA2"])
            backtest = vectorized_backtesting(
                quotes=quotes, 
                SMA1=sma1, 
                SMA2=sma2
            )
        opt_values.name = ticker
        cum_rets = backtest["Returns"].cumsum()
        cum_rets = cum_rets.apply(np.exp)
        cum_strat = backtest["Strategy"].cumsum()
        cum_strat = cum_strat.apply(np.exp)
        cum_rets.name = "CumReturns"
        cum_strat.name = "CumStrategy"
        ma_results = pd.DataFrame(quotes)
        ma_results = ma_results.dropna()
        ma_results = ma_results.join(
            [backtest[["SMA1", "SMA2", "Position"]], 
            cum_rets, cum_strat],
            how="outer"
        )
        moving_average_dict[ticker] = ma_results
        optimal_ma_value_list.append(opt_values)
    optimal_ma_values = pd.DataFrame(optimal_ma_value_list)
    optimal_ma_values = optimal_ma_values.transpose()
    return moving_average_dict, optimal_ma_values

def vectorized_backtesting(quotes, SMA1, SMA2):
    """Function performs backtesting for the simple
    moving average trading strategy for historical quotes.
    Trading strategy states a long position when the shorter
    moving average is above the longer moving average and 
    otherwise a short position. Here the returns are 
    multiplied with the position value of 1 or -1 to calculate
    the strategy performance

    :param quotes: Series of historical quotes
    :type quotes: Series
    :param SMA1: Value for calculating the shorter moving average
    :type SMA1: Integer
    :param SMA2: Value for calculating the longer moving average
    :type SMA2: Integer
    :return: Daily strategy and trading position values as well as
    daily quotes and moving averaged quotes
    :rtype: Dataframe
    """
    quotes = quotes.dropna()
    sma1 = quotes.rolling(SMA1).mean()
    sma2 = quotes.rolling(SMA2).mean()

    data = pd.DataFrame(quotes)
    data.columns = [QUOTE_COL]
    data["SMA1"] = sma1
    data["SMA2"] = sma2
    data = data.dropna()
    position_mask = np.where(
        data[SMA1_COL] > data[SMA2_COL], 1, -1
    )
    data["Position"] = position_mask
    data["Returns"] = np.log(
        data[QUOTE_COL] / data[QUOTE_COL].shift(1)
    )
    data["Strategy"] = data["Position"].shift(1) * data["Returns"]
    data = data.dropna()
    return data

def ma_optimization(quotes):
    """Function searches for optimal moving average periods
    to calculate a short term and long term moving average.
    Vectorized backtesting is used to seach for optimal 
    parameters indicating the highest out performance
    of market returns

    :param quotes: Historical stock quotes to calculate 
    moving averages
    :type quotes: Series
    :return: Series of optimal performance indicators with 
    ticker symbol as name
    :rtype: Series
    """
    total_results_list = []
    sma1 = range(20, 61, 4)
    sma2 = range(180, 281, 10)
    for SMA1, SMA2 in product(sma1, sma2):
        ma_data = vectorized_backtesting(
            quotes=quotes,
            SMA1=SMA1,
            SMA2=SMA2
        )
        performance = np.exp(
            ma_data[["Returns", "Strategy"]].sum()
        )
        market = performance["Returns"]
        strategy = performance["Strategy"]
        results = {
            "SMA1": int(SMA1),
            "SMA2": SMA2,
            "Market": market,
            "Strategy": strategy,
            "Performance": strategy - market
        }
        total_results_list.append(results)
    total_results = pd.DataFrame(total_results_list)
    total_results = total_results.sort_values(
        by="Performance",
        ascending=False
    )
    optimal_result = total_results.iloc[0, :]
    return optimal_result

def concat_dict_to_df(dict):
    """Function concats a python dictionary
    to a pandas dataframe

    :param ma_dict: Dictionary with ticker as key
    :type ma_dict: Dictionary
    :return: Dataframe with dictionary values as column
    :rtype: Dataframe
    """
    constant_cols = read_json("constant.json")["columns"]
    tick_col = constant_cols["ticker"]
    for key, value in dict.items():
        value[tick_col] = key

    df = pd.concat(
        objs=dict.values(),
        axis=0,
        ignore_index=False
        )
    return df

if __name__ == "__main__":
    parameter = read_json("parameter.json")["analysis"]
    ticker_list = read_json("inputs.json")["ticker"]
    constant_cols = read_json("constant.json")["columns"]
    SMA1_COL = constant_cols["sma1"]
    SMA2_COL = constant_cols["sma2"]
    QUOTE_COL = constant_cols["quote"]
    returns = FileAdapter().load_returns()
    closing_quotes = FileAdapter().load_closing_quotes()
    fundamentals_dict = FileAdapter().load_fundamentals()
    closing_quotes = harmonize_tickers(object=closing_quotes)
    fundamentals_dict = harmonize_tickers(object=fundamentals_dict)
    fundamentals_df = concat_dict_to_df(dict=fundamentals_dict)
    moving_averages, optimal_values = get_moving_averages(
        quote_data=closing_quotes, 
        train=False
    )
    moving_averages_df = concat_dict_to_df(dict=moving_averages)
    app = AnalysisDashboard(
        ma_data=moving_averages_df,
        ma_values = optimal_values,
        returns=returns,    
        fundamentals=fundamentals_df,
        )
    app.run()
    print("Finished")

