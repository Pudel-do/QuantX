import pandas as pd
import numpy as np
import datetime as dt
from core.finance_adapter import FinanceAdapter
from core.asset_adapter import AssetRepository
from misc.utils import read_json, get_last_business_day
from misc.schema_utils import *
from misc.utils import *

PARAMETER = read_json("parameter.json")

def read_prices(asset_id):

    sql = """
        SELECT asset_id, date, adj_close, high, low, volume
        FROM raw_prices
        WHERE asset_id = :asset_id
        ORDER BY date
        """
    
    df = read_sql(
        sql=sql,
        params={
            "asset_id": asset_id
        }
    )
    
    return df

def clean_data(df):
    df = df.fillna(method="ffill")

    return df

def calculate_price_features(df):

    #RSI
    rsi_window = PARAMETER["rsi_window"]
    delta = df["price"].diff(1)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain = pd.Series(gain, index=df.index)
    loss = pd.Series(loss, index=df.index)
    avg_gain = gain.rolling(window=rsi_window).mean()
    avg_loss = loss.rolling(window=rsi_window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi.name = "rsi"

    #MACD
    ema_fast = df["price"].ewm(span=PARAMETER["macd_fast"])
    ema_slow = df["price"].ewm(span=PARAMETER["macd_slow"])
    ema_fast = ema_fast.mean()
    ema_slow = ema_slow.mean()
    macd = ema_fast - ema_slow
    macd.name = "macd"

    #ATR
    high = df["high"]
    low = df["low"]
    high_low = high - low
    high_close = np.abs(high - df["price"].shift(1))
    low_close = np.abs(low - df["price"].shift(1))
    true_range = pd.concat(
        [high_low, high_close, low_close], 
        axis=1
    )
    true_range = true_range.max(axis=1)
    atr = true_range.rolling(window=PARAMETER["atr_window"])
    atr = atr.mean()
    atr.name = "atr"
    
    #OBV
    obv = np.where(
        df["price"] > df["price"].shift(1), 
        df["volume"], 
        np.where(
            df["price"] < df["price"].shift(1), 
            -df["volume"],
            0
        )
    )
    obv = pd.Series(obv, index=df.index)
    obv = obv.cumsum()
    obv.name = "obv"

    #ROLLING_VOLATILITY
    rets = np.log(df["price"] / df["price"].shift(1))
    vola = rets.rolling(window=PARAMETER["vol_window"])
    vola = vola.std()
    vola.name = "ret_vola"

    features = pd.concat(
        [rsi, macd, atr, obv, vola],
        axis=1,
        join="outer"
    )

    return features

def calculate_moving_average(df):
    sma_short = df["price"].rolling(PARAMETER["sma_short"]).mean()
    sma_long = df["price"].rolling(PARAMETER["sma_long"]).mean()
    df["sma_short"] = sma_short
    df["sma_long"] = sma_long

    return df

def ingest_data(data):

    sql = """
            INSERT OR IGNORE INTO processed_prices
            (asset_id, date, price, sma_short, sma_long, return, rsi, macd, atr, obv, ret_vola)
            VALUES (:asset_id, :date, :price, :sma_short, :sma_long, :return, :rsi, :macd, :atr, :obv, :ret_vola)
        """
    write_sql(sql, data)

def run():
    asset_ids = AssetRepository().get_ids()
    df_list = []
    for id in asset_ids:
        prices = read_prices(id)
        prices["price"] = prices["adj_close"]
        prices["return"] = calculate_returns(prices["price"])
        prices = calculate_moving_average(prices)
        features = calculate_price_features(prices)
        prices = prices.join(
            features,
            how="left"
        )
        df_list.append(prices)

    df_prices = pd.concat(df_list, ignore_index=True)
    ingest_data(df_prices)

if __name__ == "__main__":
    run()