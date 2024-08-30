import pandas as pd
import numpy as np
from itertools import product
from core.file_adapter import FileAdapter
from core.dashboard_adapter import AnalysisDashboard
from misc.misc import *

def get_moving_averages(quote_data, train):
    for ticker, quotes in quote_data.items():
        if train:
            train_set, test_set = ts_train_test_split(
                ts=quotes,
                train_ratio=0.75
            )
            opt_parameters = optimization(quotes=train_set)
            sma1 = int(opt_parameters.loc[SMA1_COL])
            sma2 = int(opt_parameters.loc[SMA2_COL])
            backtest = vectorized_backtesting(
                quotes=test_set, 
                SMA1=sma1, 
                SMA2=sma2
            )
        else:
            opt_parameters = optimization(quotes=quotes)
            sma1 = int(opt_parameters.loc[SMA1_COL])
            sma2 = int(opt_parameters.loc[SMA2_COL])
            backtest = vectorized_backtesting(
                quotes=quotes, 
                SMA1=sma1, 
                SMA2=sma2
            )
        backtest = backtest.drop(columns=QUOTE_COL)
        cum_rets = backtest["Returns"].cumsum()
        cum_rets = cum_rets.apply(np.exp)
        cum_strat = backtest["Strategy"].cumsum()
        cum_strat = cum_strat.apply(np.exp)
        cum_rets.name = "CumReturns"
        cum_strat.name = "CumStrategy"
        ma_results = pd.DataFrame(quotes)
        ma_results = ma_results.dropna()
        ma_results = ma_results.join(
            [backtest, cum_rets, cum_strat],
            how="outer"
        )
        print("break")

def vectorized_backtesting(quotes, SMA1, SMA2):
    quotes = quotes.dropna()
    sma1 = quotes.rolling(SMA1).mean()
    sma2 = quotes.rolling(SMA2).mean()

    data = pd.DataFrame(quotes)
    data.columns = [QUOTE_COL]
    data[SMA1_COL] = sma1
    data[SMA2_COL] = sma2
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

def optimization(quotes):
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
            SMA1_COL: SMA1,
            SMA2_COL: SMA2,
            "MARKET": market,
            "STRATEGY": strategy,
            "OUT": strategy - market
        }
        total_results_list.append(results)
    total_results = pd.DataFrame(total_results_list)
    total_results = total_results.sort_values(
        by="OUT",
        ascending=False
    )
    optimal_result = total_results.iloc[0, :]
    optimal_result = optimal_result.transpose()
    return optimal_result

def get_moving_average(quotes, ma_days):
    """Function calculates simple moving averages for given
    amount of short period and long period days in inputs.json file. 
    All missing values are removed for calculation for the underlying 
    quote data

    :param quotes: Quotes data for given stocks
    :type quotes: Dataframe
    :param ma_days: Days for calculating moving averages
    :type ma_days: List
    :return: Dictionary with ticker symbols as key and
    quote and moving averages as value
    :rtype: Dictionary
    """
    ma_dict = {}
    days_short = ma_days[0]
    days_long = ma_days[1]
    quotes = quotes.dropna()
    for name, values in quotes.items():
        values.name = "Quote"
        ma_short = values.rolling(
            window=days_short
        ).mean()
        ma_short.name = f"MovingAverage{days_short}"

        ma_long = values.rolling(
            window=days_long
        ).mean()
        ma_long.name = f"MovingAverage{days_long}"

        ma_df = pd.concat(
            objs=[values, ma_short, ma_long],
            axis=1,
            join="inner",
            ignore_index=False
        )
        ma_dict[name] = ma_df
    return ma_dict

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

    ma_test = get_moving_averages(quote_data=closing_quotes, train=False)

    # moving_average_dict = get_moving_average(
    #     quotes=closing_quotes, 
    #     ma_days=ma_days
    #     )
    # moving_average_df = concat_dict_to_df(
    #     dict=moving_average_dict
    #     )
    # fundamentals_df = concat_dict_to_df(
    #     dict=fundamentals_dict
    # )
    # app = AnalysisDashboard(
    #     ma_data=moving_average_df,
    #     returns=returns,
    #     fundamentals=fundamentals_df,
    #     )
    # app.run()
    # print("Finished")

