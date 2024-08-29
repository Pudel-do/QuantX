
import pandas as pd
import numpy as np
from core.file_adapter import FileAdapter
from core.dashboard_adapter import AnalysisDashboard
from misc.misc import *

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
    fundamental_list = read_json("constant.json")["fundamentals"]["measures"]
    ma_days = parameter["ma_days"]
    returns = FileAdapter().load_returns()
    closing_quotes = FileAdapter().load_closing_quotes()
    fundamentals_dict = FileAdapter().load_fundamentals()
    closing_quotes = harmonize_tickers(object=closing_quotes)
    fundamentals_dict = harmonize_tickers(object=fundamentals_dict)
    moving_average_dict = get_moving_average(
        quotes=closing_quotes, 
        ma_days=ma_days
        )
    moving_average_df = concat_dict_to_df(
        dict=moving_average_dict
        )
    fundamentals_df = concat_dict_to_df(
        dict=fundamentals_dict
    )
    app = AnalysisDashboard(
        tickers=ticker_list,
        ma_data=moving_average_df,
        returns=returns,
        fundamentals=fundamentals_df,
        fundamental_list=fundamental_list
        )
    app.run()
    print("Finished")

