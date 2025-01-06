import pandas as pd
import numpy as np
from mttkinter import mtTkinter as tk
from core.file_adapter import FileAdapter
from misc.misc import *
from core.models import OneStepLSTM, MultiStepLSTM, ArimaModel
from core.finance_adapter import FinanceAdapter
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error

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
        processed_data_dict[tick] = data

    return processed_data_dict

def feature_engineering(stock_dict):
    """Function calculates technical indicators as features
    for prediciton models. All indicator must be defined in 
    constant.json. 

    :param data_dict: Daily trading data
    :type data_dict: Dictionary
    :return: Set of technical indicators
    for each ticker
    :rtype: Dictionary
    """
    model_data = {}
    for tick, data in stock_dict.items():
        quotes = data[PARAMETER["target_col"]]
        #RSI
        rsi_window = PARAMETER["rsi_window"]
        delta = quotes.diff(1)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        gain = pd.Series(gain, index=data.index)
        loss = pd.Series(loss, index=data.index)
        avg_gain = gain.rolling(window=rsi_window).mean()
        avg_loss = loss.rolling(window=rsi_window).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        rsi.name = CONST_COLS["rsi"]

        #MACD
        ema_fast = quotes.ewm(span=PARAMETER["macd_fast"])
        ema_slow = quotes.ewm(span=PARAMETER["macd_slow"])
        ema_fast = ema_fast.mean()
        ema_slow = ema_slow.mean()
        macd = ema_fast - ema_slow
        macd.name = CONST_COLS["macd"]

        #ATR
        high = data[CONST_COLS["high"]]
        low = data[CONST_COLS["low"]]
        high_low = high - low
        high_close = np.abs(high - quotes.shift(1))
        low_close = np.abs(low - quotes.shift(1))
        true_range = pd.concat(
            [high_low, high_close, low_close], 
            axis=1
        )
        true_range = true_range.max(axis=1)
        atr = true_range.rolling(window=PARAMETER["atr_window"])
        atr = atr.mean()
        atr.name = CONST_COLS["atr"]
        
        #OBV
        obv = np.where(
            quotes > quotes.shift(1), 
            data[CONST_COLS["volume"]], 
            np.where(
                quotes < quotes.shift(1), 
                -data[CONST_COLS["volume"]],
                0
            )
        )
        obv = pd.Series(obv, index=data.index)
        obv = obv.cumsum()
        obv.name = CONST_COLS["obv"]

        #ROLLING_VOLATILITY
        rets = np.log(quotes / quotes.shift(1))
        vola = rets.rolling(window=PARAMETER["vol_window"])
        vola = vola.std()
        vola.name = CONST_COLS["ret_vola"]

        target = pd.DataFrame(quotes)
        features = pd.concat(
            [rsi, macd, atr, obv, vola],
            axis=1,
            join="outer"
        )
        features = features[PARAMETER["feature_cols"]]
        data = target.join(
            features,
            how="inner"
        )
        data = data.dropna()
        model_data[tick] = data
    return model_data
        
def model_building(model_data, models):
    """Function builds, train and evaluates given
    forecast model. Finally the model is saved to 
    model and ticker directory

    :param model_data: Preprocessed data for model building
    :type modnel_data:  Dictionary
    :param models: Prediction models to build
    :type models: List
    :return: None
    :rtype: None
    """
    for tick, data in model_data.items():
        for model in models:
            model.init_data(
                data=data,
                ticker=tick
            )
            model.preprocess_data()
            model.build_model()
            if PARAMETER["use_hp_tuning"]:
                model.hyperparameter_tuning()
            else:
                pass
            model.train()
            model.evaluate()
            FileAdapter().save_model(model=model)
            del model
    return None

def model_backtesting(tickers):
    """Function merges closing quotes up to the current last
    business day for given tickers with the prediction values 
    of the last current prediction models separated by model types.
    Furthermore, ifferent performance measures rom the out-of-sample 
    prediction are calculated for all available model types for each ticker.
    The performance measures are directly defined in the function and
    must be defined or adjusted in this place.

    :param tickers: Stock tickers for backtesting
    :type tickers: List
    :return: Daily prediction and backtested prediction values as well as
    prediction performance measure for backtesting period and list with
    included models
    :rtype: Dictionary, Dictionary, List
    """
    backtest_dict = {}
    validation_dict = {}
    model_list = []
    for tick in tickers:
        trade_data = FinanceAdapter(tick=tick).get_trade_data(
            start=PARAMETER["model_start"]
        )
        trade_data = rename_yfcolumns(data=trade_data)
        closing_quotes = trade_data[PARAMETER["quote_id"]]
        closing_quotes.name = CONST_COLS["quote"]
        closing_quotes = pd.DataFrame(closing_quotes)
        model_ids = get_latest_modelid(
            tick=tick, 
            model_type=None
        )
        backtest = pd.DataFrame(closing_quotes)
        backtest_validation = pd.DataFrame()
        for model_id in model_ids:
            model = FileAdapter().load_model(
                ticker=tick,
                model_id=model_id
            )
            model_type = model_id.split("_")[-1]
            model_type = model_type.split(".")[0]
            if model_type not in model_list:
                model_list.append(model_type)
            prediction = model.predict()
            prediction.name = model_type
            backtest = backtest.join(
                prediction,
                how="right",
            )
            validation = closing_quotes.join(
                prediction,
                how="inner"
            )
            if not validation.empty:
                actual = validation[CONST_COLS["quote"]]
                pred = validation[model_type]
                rmse = root_mean_squared_error(actual, pred)
                mape = mean_absolute_percentage_error(actual, pred)
                mae = mean_absolute_error(actual, pred)
                backtest_validation.loc[CONST_COLS["rmse"], model_type] = rmse
                backtest_validation.loc[CONST_COLS["mae"], model_type] = mae
                backtest_validation.loc[CONST_COLS["mape"], model_type] = mape
            else:
                break
        backtest_dict[tick] = backtest
        validation_dict[tick] = backtest_validation
    return backtest_dict, validation_dict, model_list


if __name__ == "__main__":
    CONST_COLS = read_json("constant.json")["columns"]
    CONST_DATA = read_json("constant.json")["datamodel"]
    PARAMETER = read_json("parameter.json")
    closing_quotes = FileAdapter().load_dataframe(
        path=CONST_DATA["raw_data_dir"],
        file_name=CONST_DATA["quotes_file"]
    )
    daily_trading_data = FileAdapter().load_object(
        path=CONST_DATA["feature_dir"],
        file_name=CONST_DATA["daily_trading_data_file"]
    )
    raw_tick_dict = merge_features(
        quotes=closing_quotes, 
        features=daily_trading_data
    )
    processed_tick_dict = data_cleaning(
        data_dict=raw_tick_dict
    )
    model_data_dict = feature_engineering(
        stock_dict=processed_tick_dict
    )
    model_data_dict, tickers = harmonize_tickers(
        object=model_data_dict
    )
    FileAdapter().save_object(
        obj=model_data_dict,
        path=CONST_DATA["raw_data_dir"],
        file_name=CONST_DATA["model_data_file"]
    )
    models = [
        MultiStepLSTM(),
    ]
    if PARAMETER["use_model_training"]:
        model_building(
            model_data=model_data_dict, 
            models=models
        )
    backtest, validation, models = model_backtesting(
        tickers=tickers
    )
    FileAdapter().save_object(
        obj=backtest,
        path=CONST_DATA["processed_data_dir"],
        file_name=CONST_DATA["backtest_model_file"]
    )
    FileAdapter().save_object(
        obj=validation,
        path=CONST_DATA["processed_data_dir"],
        file_name=CONST_DATA["validation_model_file"]
    )
    FileAdapter().save_object(
        obj=models,
        path=CONST_DATA["processed_data_dir"],
        file_name=CONST_DATA["model_list"]
    )