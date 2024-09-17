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
    last_bday = last_bday.normalize()
    last_bday = last_bday.strftime(format="%Y-%m-%d")
    return last_bday

def harmonize_tickers(object):
    """Functions harmonizes ticker symbols in input object
    with base tickers in parameter.json for dataframes and
    dictionary. If input object is different type, the 
    function returns the raw input object

    :param object: Quotes or fundamental data
    :type object: Dataframe, Dictionary
    :return: Input object adjusted for base tickers
    :rtype: Dataframe, Dictionary
    """
    base_tickers = read_json("parameter.json")["ticker"]
    object_clean = object.copy()
    overlap_tickers = []
    if isinstance(object, pd.DataFrame):
        for col in object.columns:
            if col not in base_tickers:
                object_clean.drop(
                    columns=col, 
                    inplace=True
                    )
            else:
                overlap_tickers.append(col)
        return object_clean, overlap_tickers
    elif isinstance(object, dict):
        for key in object:
            if key not in base_tickers:
                object_clean.pop(key)
            else:
                overlap_tickers.append(key)
        return object_clean, overlap_tickers
    else:
        return object, base_tickers
    
def ts_train_test_split(ts, train_ratio):
    """Function splits time series into
    train and test set

    :param ts: Time series or array to split
    :type ts: Pandas series or array
    :param train_ratio: Ratio for training sample
    :type train_ratio: float
    :return: Train and test set
    :rtype: Pandas series
    """
    split_index = int(len(ts) * train_ratio)
    train = ts[:split_index]
    test = ts[split_index:]
    return train, test

def get_latest_modelid(tick, model_type):
    """Function filters in model directory for the most actual 
    forecas model separated by the respective ticker symbol and
    forecast model type. If model_type is not None, the model
    of model_type is directly selected and the most actual model ID
    is returned

    :param tick: Ticker to search model for
    :type tick: String
    :param model_type: Type of forecast model (e.g. ARIMA, LSTM)
    for direct selection
    :type model_type: String
    :return: Most actual model ID for respective ticker and model tyoe
    :rtype: List
    """
    datamodel = read_json("constant.json")["datamodel"]
    models_dir = datamodel["models_dir"]
    model_dir = datamodel["model"]
    models_path = os.path.join(models_dir, tick, model_dir)
    models = os.listdir(models_path)
    model_types = []
    model_dates = []
    model_dict = {}
    for model in models:
        model_split = model.split("_")
        date = model_split[0]
        model_name = model_split[1]
        if not model_name in model_types:
            model_dict[model_name] = [date]
            model_types.append(model_name)
        else:
            model_dict.get(model_name).append(date)
    model_ids = []
    for key, value in model_dict.items():
        if model_type is None:
            latest_date = max(value)
            model_id = f"{latest_date}_{key}"
            model_ids.append(model_id)
        else:
            if model_type in key:
                latest_date = max(value)
                model_id = f"{latest_date}_{key}"
                model_ids.append(model_id)
    return model_ids

    

    