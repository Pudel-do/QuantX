import pandas as pd
import numpy as np
from core.file_adapter import FileAdapter
from misc.misc import *
from core.models import OneStepLSTM, MultiStepLSTM
from core.finance_adapter import FinanceAdapter
from core.dashboard_adapter import ModelBackTesting
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

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

def feature_engineering(data_dict):
    """Function calculates features for model building
    Output should display processed data for model building
    """
    model_features = {}
    for tick, data in data_dict.items():
        quotes = data[CONST_COLS["quote"]]
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

        quotes = pd.DataFrame(quotes)
        features = quotes.join(
            [rsi, macd, atr, obv, vola],
            how="inner"
        )
        features = features[PARAMETER["feature_cols"]]
        features = features.dropna()
        model_features[tick] = features
    return model_features
        
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
        closing_quotes = trade_data[CONST_COLS["quote_id"]]
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
            actual = validation[CONST_COLS["quote"]]
            pred = validation[model_type]
            rmse = np.sqrt(mean_squared_error(actual, pred))
            mape = mean_absolute_percentage_error(actual, pred)
            mae = mean_absolute_error(actual, pred)
            backtest_validation.loc[CONST_COLS["rmse"], model_type] = rmse
            backtest_validation.loc[CONST_COLS["mae"], model_type] = mae
            backtest_validation.loc[CONST_COLS["mape"], model_type] = mape
        backtest_validation.index.rename(name=CONST_COLS["measures"], inplace=True)
        backtest_validation.reset_index(inplace=True)
        backtest_dict[tick] = backtest
        validation_dict[tick] = backtest_validation
    return backtest_dict, validation_dict, model_list


if __name__ == "__main__":
    CONST_COLS = read_json("constant.json")["columns"]
    PARAMETER = read_json("parameter.json")
    closing_quotes = FileAdapter().load_closing_quotes()
    daily_trading_data = FileAdapter().load_trading_data()
    raw_data_dict = merge_features(
        quotes=closing_quotes, 
        features=daily_trading_data
    )
    processed_data_dict = data_cleaning(data_dict=raw_data_dict)
    model_data_dict = feature_engineering(processed_data_dict)
    model_data_dict, tickers = harmonize_tickers(model_data_dict)
    FileAdapter().save_model_data(model_data=model_data_dict)
    models = [
        OneStepLSTM()
    ]
    if PARAMETER["use_model_training"]:
        model_building(
            model_data=model_data_dict, 
            models=models
        )
    backtest_dict, validation_dict, models = model_backtesting(tickers=tickers)
    app = ModelBackTesting(
        backtest_dict=backtest_dict,
        validation_dict=validation_dict,
        model_list=models
    )
    app.run(debug=True)