import pandas as pd
import json
import os
import logging
from core import logging_config
from datetime import datetime

def read_json(file_name):
    """
    :param file_name: json file name to read
    :type file_name: string
    :return: Loaded json file
    :rtype: dictionary
    """
    try:
        with open (os.path.join(os.getcwd(), "config", file_name)) as file:
            config = json.load(file)
        return config
    except FileNotFoundError:
        logging.error(f"File {file_name} not found")
        return None

def get_business_day(date):
    """
    :param date: Base date for calculating the previous business day
    :type date: string or date object
    :return: Previous business day for given date. 
    If param date is still a business day, the function returns param date
    :rtype: string
    """
    if not isinstance(date, datetime):
        date = pd.to_datetime(date)
    is_bday = pd.bdate_range(start=date, end=date).shape[0] > 0
    if not is_bday:
        bday = date - pd.offsets.BDay(1)
        bday = bday.strftime(format="%Y-%m-%d")
    else:
        bday = date.strftime(format="%Y-%m-%d")
    return bday

def get_last_business_day():
    """Function calculates the last business day
    compared to actual date

    :return: Last busines day based on actual date
    :rtype: String
    """
    now = datetime.now()
    last_bday = now - pd.offsets.BDay(1)
    last_bday = last_bday.strftime(format="%Y-%m-%d")
    return last_bday

def harmonize_tickers(object):
    """Functions harmonizes ticker symbols in input object
    with base tickers in inputs.json for dataframes and
    dictionary. If input object is different type, the 
    function returns the raw input object

    :param object: Quotes or fundamental data
    :type object: Dataframe, Dictionary
    :return: Input object adjusted for base tickers
    :rtype: Dataframe, Dictionary
    """
    base_tickers = read_json("inputs.json")["ticker"]
    object_clean = object.copy()
    if isinstance(object, pd.DataFrame):
        for col in object.columns:
            if col not in base_tickers:
                object_clean.drop(
                    columns=col, 
                    inplace=True
                    )
            else:
                pass
        return object_clean
    elif isinstance(object, dict):
        for key in object:
            if key not in base_tickers:
                object_clean.pop(key)
            else:
                pass
        return object_clean 
    else:
        return object
    
def ts_train_test_split(ts, train_ratio):
    """Function splits time series into
    train and test set

    :param ts: Time series to split
    :type ts: Pandas series
    :param train_ratio: Ratio for training sample
    :type train_ratio: float
    :return: Train and test set
    :rtype: Pandas series
    """
    split_index = int(len(ts) * train_ratio)
    train = ts.iloc[:split_index]
    test = ts.iloc[split_index:]
    return train, test