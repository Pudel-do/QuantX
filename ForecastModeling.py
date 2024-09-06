import pandas as pd
import numpy as np
from core.file_adapter import FileAdapter
from misc.misc import *



def merge_features(quotes, features):
    """Function merges daily quote data with
    daily feature data from feature directory.
    Feature data is filtered for defined feature
    columns in parameter.json file

    :param quotes: Daily stock quotes for different tickers
    :type quotes: Dataframe
    :param features: Daily feature data for different tickers
    :type features: Dictionary
    :return: Merged quotes and features on daily base
    :rtype: Dictionary
    """
    data_dict = {}
    for tick, ser in quotes.items():
        tick_features = features[tick]
        tick_features = tick_features[PARAMETER["feature_cols"]]
        df_quotes = pd.DataFrame(data=ser)
        df_quotes = df_quotes.rename(
            columns={tick: CONST_COLS["quote"]}
        )
        merged_data = df_quotes.join(
            tick_features,
            how="left"
        )
        data_dict[tick] = merged_data

    return data_dict

def data_cleaning(data_dict):
    """Function removes missing values for raw model
    data and filters for defined model start period
    in parameter.json

    :param data_dict: Raw model data
    :type data_dict: Dictionary
    :return: Processed model data
    :rtype: Dictionary
    """
    processed_data_dict = {}
    for tick, data in data_dict.items():
        period_mask = data.index >= PARAMETER["model_start"]
        data = data[period_mask]
        data = data.dropna()
        data[CONST_COLS["volume"]] = data[CONST_COLS["volume"]].astype(int)
        processed_data_dict[tick] = data

    return processed_data_dict

def feature_engineering():
    """Function calculates features for model building
    Output should display processed data for model building
    """
    pass

if __name__ == "__main__":
    CONST_COLS = read_json("constant.json")["columns"]
    PARAMETER = read_json("parameter.json")
    closing_quotes = FileAdapter().load_closing_quotes()
    daily_trading_data = FileAdapter().load_trading_data()
    raw_data_dict = merge_features(quotes=closing_quotes, features=daily_trading_data)
    processed_data_dict = data_cleaning(data_dict=raw_data_dict)
    processed_data_dict = harmonize_tickers(processed_data_dict)
    print("break")