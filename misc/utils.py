import pandas as pd
import numpy as np
import json
import os
import logging
import io
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
from core import logging_config
from datetime import datetime
matplotlib.use('Agg') 

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
    
def get_list_intersection(*lists):
    """Function returns intersection values of
    entries in given lists

    :return: List intersection values
    :rtype: List
    """
    sets = map(set, lists)
    intersection = set.intersection(*sets)
    intersection = list(intersection)
    return intersection
    
    
def calculate_returns(quotes):
    """Function calculates log returns for
    given quotes and time range

    :param quotes: Quotes from stocks
    :type quotes: Dataframe
    :return: Stock returns
    :rtype: Dataframe
    """
    rets = np.log(quotes / quotes.shift(1))
    rets = rets.iloc[1:]
    return rets

def cumulate_returns(returns):
    """Function calculates the cumulative and
    exponential portfolio returns

    :param returns: Daily portfolio returns
    :type returns: Dataframe
    :return: Cumulative portfolio returns
    :rtype: Dataframe
    """
    cum_returns = pd.DataFrame()
    for name, values in returns.items():
        try:
            cum_rets = values.cumsum().apply(np.exp)
            cum_rets = cum_rets * 1000
            cum_returns[name] = cum_rets
        except:
            logging.error(f"Return cumulation failed for portfolio type {name}")
            cum_returns[name] = values
    return cum_returns
    
def rename_yfcolumns(data):
    """Function renames dataframe columns
    by constant column names in constant.json

    :param data: Data for column name adjusting
    :type data: Dataframe
    :return: Dataframe adjusted by column names
    :rtype: 
    """
    CONST_COLS = read_json("constant.json")["columns"]
    try:
        rename_dict = {
            "Adj Close": CONST_COLS["adj_close"],
            "Close": CONST_COLS["close"],
            "Open": CONST_COLS["open"],
            "High": CONST_COLS["high"],
            "Low": CONST_COLS["low"],
            "Volume": CONST_COLS["volume"],
        }
    except ValueError:
        logging.warning("Constant columns not defined in json file")
        rename_dict = {}

    data_adj = data.rename(columns=rename_dict)
    return data_adj

def get_business_day(date):
    """
    :param date: Base date for calculating the previous business day
    :type date: string or date object
    :return: Previous business day for given date. 
    If param date is still a business day, the function returns param date
    :rtype: String
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

def calc_annualized_mean_return(returns):
    """Function calculates annualized mean return for daily 
    return series

    :param returns: Daily stock returns
    :type returns: Series
    :return: Annualized mean return
    :rtype: Float
    """
    ann_mean_ret = returns.mean() * 252
    return ann_mean_ret

def calc_total_return(returns):
    """Function calculates total return for daily return series

    :param returns: Daily stock returns
    :type returns: Series
    :return: Total stock return
    :rtype: Float
    """
    total_ret = (np.exp(returns.sum()) - 1)
    return total_ret

def calc_annualized_vola(returns):
    """Function calculates annualized volatility for daily
    return series

    :param returns: Daily stock returns
    :type returns: Series
    :return: Annualized volatililty
    :rtype: Float
    """
    ann_vola = np.sqrt(returns.std() * 252)
    return ann_vola


def harmonize_tickers(object):
    """Functions harmonizes ticker symbols in input object
    with base tickers in parameter.json for dataframes and
    dictionary. If input object is different type, the 
    function returns the raw input object

    :param object: Data to harmonize
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
    if not os.path.exists(models_path):
        logging.warning(f"Model for ticker {tick} does not exist")
        return None
    else:
        models = os.listdir(models_path)
    model_types = []
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
        key_split = key.split(".")
        key_split = key_split[0]
        if model_type is None:
            latest_date = max(value)
            model_id = f"{latest_date}_{key_split}"
            model_ids.append(model_id)
        else:
            if model_type in key_split:
                latest_date = max(value)
                model_id = f"{latest_date}_{key_split}"
                return model_id
            else:
                logging.warning(f"Model type {model_type} not in model folder")
                return None
    return model_ids
            
def get_future_returns(tickers, rets):
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
    from core.file_adapter import FileAdapter
    params = read_json("parameter.json")
    return_list = []
    for tick in tickers:
        model_id = get_latest_modelid(
            tick=tick, 
            model_type=params["model"]
        )
        model = FileAdapter().load_model(
            ticker=tick,
            model_id=model_id
        )
        if model_id is None:
            model.init_data(
                data=rets,
                ticker=tick
            )
        quote_prediction = model.predict()
        returns = calculate_returns(quotes=quote_prediction)
        returns.name = tick
        return_list.append(returns)

    future_returns = pd.concat(return_list, axis=1)
    return future_returns

def get_log_path(ticker, model_id, log_key):
    """Function builds path for saving model logs.
    Underlying directory elements are defined in 
    constant.json

    :return: Path for saving model logs
    :rtype: String
    """
    data_model = read_json("constant.json")["datamodel"]
    models_dir = data_model["models_dir"]
    log_dir = data_model[log_key]
    path = os.path.join(
        os.getcwd(), 
        models_dir,
        ticker,
        log_dir, 
        model_id
    )
    return path

def create_in_pred_fig(ticker, target, y_pred, rmse, facts):
    """Function builds matplot figure to visualize
    prediction performance on the test set against
    target values from the same set. In addition, 
    the RMSE is calculated viszualized as performance
    measure

    :param ticker: Ticker for predicted stock
    :type ticker: String
    :param target: Real target values from test set
    :type target: Array
    :param y_pred: Prediciton values from test set
    :type y_pred: Array
    :param rmse: Root mean squared error from test set performance
    :type rmse: Float
    :return: Line chart figure
    :rtype: Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12,2))
    ax.plot(target, label="Real Values", color='blue')
    ax.plot(y_pred, label="Predicted Values", color='red')
    ax.legend(loc="best")
    ax.set_title(f"In-Sample prediction for {ticker} with RMSE of {rmse}")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Value")
    ax.text(
        1.02, 1.0,
        facts,
        transform=ax.transAxes,
        fontsize=10,
        ha='left',
        va='top',
    )
    plt.tight_layout()
    return fig


def plot_to_image(figure):
    """Function saves figure as tensorflow image

    :param figure: Figure to save
    :type figure: Matplotlib figure
    :return: Tensorflow image
    :rtype: Tensorflow image
    """
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image
    

    