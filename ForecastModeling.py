import pandas as pd
import numpy as np
from core.file_adapter import FileAdapter
from misc.misc import *
from core.models import lstm
from core.finance_adapter import FinanceAdapter
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
        processed_data_dict[tick] = data

    return processed_data_dict

def feature_engineering():
    """Function calculates features for model building
    Output should display processed data for model building
    """
    pass

def model_building(model_data):
    """Function builds, train and evaluates given
    forecast model. Finally the model is saved to 
    model and ticker directory

    :param model_data: Preprocessed data for model building
    :type modnel_data:  Dictionary
    :return: None
    :rtype: None
    """
    for tick, data in model_data.items():
        model = lstm(ticker=tick)
        model.load_data(data=data)
        model.preprocess_data()
        model.build_model()
        model.train()
        model.evaluate()
        FileAdapter().save_model(model=model)
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
    measures_dict = {}
    model_list = []
    for tick in tickers:
        closing_quotes = FinanceAdapter(tick=tick).get_trade_data(
            start=PARAMETER["model_start"]
        )
        closing_quotes = closing_quotes[CONST_COLS["quote_id"]]
        closing_quotes.name = CONST_COLS["quote"]
        closing_quotes = pd.DataFrame(closing_quotes)
        model_ids = get_latest_modelid(
            tick=tick, 
            model_type=None
        )
        measures_df = pd.DataFrame()
        backtest_df = pd.DataFrame(closing_quotes)
        for model_id in model_ids:
            model = FileAdapter().load_model(
                ticker=tick,
                model_id=model_id
            )
            model_type = model_id.split("_")[-1]
            model_type = model_type.split(".")[0]
            if model_type not in model_list:
                model_list.append(model_type)
            pred_days = PARAMETER["prediction_days"]
            prediction = model.predict(pred_days=pred_days)
            prediction.name = model_type
            backtest_df = backtest_df.join(
                prediction,
                how="outer",
            )
            validation_data = backtest_df.dropna()
            actual = validation_data[CONST_COLS["quote"]]
            pred = validation_data[model_type]
            rmse = np.sqrt(mean_squared_error(actual, pred))
            mape = mean_absolute_percentage_error(actual, pred)
            mae = mean_absolute_error(actual, pred)
            df_index = [
                CONST_COLS["rmse"],
                CONST_COLS["mae"],
                CONST_COLS["mape"], 
            ]
            measures_df.loc[df_index[0], model_type] = rmse
            measures_df.loc[df_index[1], model_type] = mae
            measures_df.loc[df_index[2], model_type] = mape

        backtest_dict[tick] = backtest_df
        measures_dict[tick] = measures_df
    return backtest_dict, measures_dict, model_list


if __name__ == "__main__":
    CONST_COLS = read_json("constant.json")["columns"]
    PARAMETER = read_json("parameter.json")
    closing_quotes = FileAdapter().load_closing_quotes()
    daily_trading_data = FileAdapter().load_trading_data()
    raw_data_dict = merge_features(quotes=closing_quotes, features=daily_trading_data)
    processed_data_dict = data_cleaning(data_dict=raw_data_dict)
    processed_data_dict, tickers = harmonize_tickers(processed_data_dict)
    FileAdapter().save_model_data(model_data=processed_data_dict)
    if PARAMETER["use_model_training"]:
        model_building(model_data=processed_data_dict)
    backtesting, measures, models = model_backtesting(tickers=tickers)
    print("break")