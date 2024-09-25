import pandas as pd
import numpy as np
from itertools import product
from core.file_adapter import FileAdapter
from core.dashboard_adapter import AnalysisDashboard
from misc.misc import *

def get_moving_averages(quote_data, use_train):
    """Function calculates moving averages as well as
    historical trading strategy returns and determines
    the strategy outperformance by comparing strategy
    returns with market returns for the same stock

    :param quote_data: Historical stock quote data
    :type quote_data: Series
    :param use_train: Parameter to control for training and testing 
    samples when calculating the trading strategy performance and
    the underlying moving average parameters
    :type use_train: Boolean
    :return: Stock quotes extended by moving average quotes and market
    and strategy returns as well as optimal moving average parameters
    :rtype: Dictionary, Series
    """
    moving_average_dict = {}
    optimal_ma_value_list = []
    for ticker, quotes in quote_data.items():
        quotes.name = CONST_COLS["quote"]
        if use_train:
            train_set, test_set = ts_train_test_split(
                ts=quotes,
                train_ratio=PARAMETER["ma_train_ratio"]
            )
            opt_values = ma_optimization(quotes=train_set)
            sma1 = int(opt_values.loc[CONST_COLS["sma1"]])
            sma2 = int(opt_values.loc[CONST_COLS["sma2"]])
            backtest = vectorized_backtesting(
                quotes=test_set, 
                SMA1=sma1, 
                SMA2=sma2
            )
        else:
            opt_values = ma_optimization(quotes=quotes)
            sma1 = int(opt_values.loc[CONST_COLS["sma1"]])
            sma2 = int(opt_values.loc[CONST_COLS["sma2"]])
            backtest = vectorized_backtesting(
                quotes=quotes, 
                SMA1=sma1, 
                SMA2=sma2
            )
        opt_values.name = ticker
        cum_rets = backtest[CONST_COLS["returns"]].cumsum()
        cum_rets = cum_rets.apply(np.exp)
        cum_strat = backtest[CONST_COLS["strategy"]].cumsum()
        cum_strat = cum_strat.apply(np.exp)
        cum_rets.name = CONST_COLS["cumreturns"]
        cum_strat.name = CONST_COLS["cumstrategy"]
        ma_results = pd.DataFrame(quotes)
        ma_results = ma_results.dropna()
        ma_results = ma_results.join(
            [backtest[
                [CONST_COLS["sma1"], 
                 CONST_COLS["sma2"], 
                 CONST_COLS["position"]
                 ]], 
            cum_rets, cum_strat
            ],
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
    data.columns = [CONST_COLS["quote"]]
    data[CONST_COLS["sma1"]] = sma1
    data[CONST_COLS["sma2"]] = sma2
    data = data.dropna()
    position_mask = np.where(
        data[CONST_COLS["sma1"]] > data[CONST_COLS["sma2"]], 
        1, -1
    )
    data[CONST_COLS["position"]] = position_mask
    data[CONST_COLS["returns"]] = np.log(
        data[CONST_COLS["quote"]] / \
            data[CONST_COLS["quote"]].shift(1)
    )
    strategy = data[CONST_COLS["position"]].shift(1) * \
                	data[CONST_COLS["returns"]]
    data[CONST_COLS["strategy"]] = strategy
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
            ma_data[[CONST_COLS["returns"], 
                     CONST_COLS["strategy"]]].sum()
        )
        market = performance[CONST_COLS["returns"]]
        strategy = performance[CONST_COLS["strategy"]]
        results = {
            CONST_COLS["sma1"]: int(SMA1),
            CONST_COLS["sma2"]: int(SMA2),
            CONST_COLS["market"]: market,
            CONST_COLS["strategy"]: strategy,
            CONST_COLS["performance"]: strategy - market
        }
        total_results_list.append(results)
    total_results = pd.DataFrame(total_results_list)
    total_results = total_results.sort_values(
        by=CONST_COLS["performance"],
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
    constant_cols = read_json("constant.json")["columns"]
    CONST_COLS = read_json("constant.json")["columns"]
    PARAMETER = read_json("parameter.json")
    use_ma_training = PARAMETER["use_ma_training"]
    returns = FileAdapter().load_returns()
    closing_quotes = FileAdapter().load_closing_quotes()
    fundamentals_dict = FileAdapter().load_fundamentals()
    closing_quotes, _ = harmonize_tickers(object=closing_quotes)
    fundamentals_dict, _ = harmonize_tickers(object=fundamentals_dict)
    fundamentals_df = concat_dict_to_df(dict=fundamentals_dict)
    moving_averages, optimal_values = get_moving_averages(
        quote_data=closing_quotes, 
        use_train=use_ma_training
    )
    moving_averages_df = concat_dict_to_df(dict=moving_averages)
    app = AnalysisDashboard(
        ma_data=moving_averages_df,
        ma_values = optimal_values,
        returns=returns,    
        fundamentals=fundamentals_df,
        )
    app.run(debug=True)