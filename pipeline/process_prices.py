import pandas as pd
import numpy as np
import datetime as dt
from core.finance_adapter import FinanceAdapter
from core.asset_adapter import AssetRepository
from misc.utils import read_json
from misc.schema_utils import *
from misc.utils import *

PARAMETER = read_json("parameter.json")

def read_prices(asset_id):

    query = """
        SELECT asset_id, date, adj_close, high, low, volume
        FROM raw_prices
        WHERE asset_id = :asset_id
        ORDER BY date
        """
    
    df = read_sql(
        query=query,
        params={
            "asset_id": asset_id
        }
    )
    
    return df

def clean_data(df):
    df = df.fillna(method="ffill")

    return df

def build_price_features(df):

    sma_short = df["price"].rolling(PARAMETER["ma_short"]).mean()
    sma_long = df["price"].rolling(PARAMETER["ma_long"]).mean()
    df["sma_short"] = sma_short
    df["sma_long"] = sma_long

    ema_short = df["price"].ewm(PARAMETER["ma_short"]).mean()
    ema_long = df["price"].ewm(PARAMETER["ma_long"]).mean()
    df["ema_short"] = ema_short
    df["ema_long"] = ema_long

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
    df["rsi"] = rsi

    ema_fast = df["price"].ewm(span=PARAMETER["macd_fast"])
    ema_slow = df["price"].ewm(span=PARAMETER["macd_slow"])
    ema_fast = ema_fast.mean()
    ema_slow = ema_slow.mean()
    macd = ema_fast - ema_slow
    df["macd"] = macd

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
    df["atr"] = atr
    
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
    df["obv"] = obv

    features = df[
        ["asset_id", "date", "price", 
        "sma_short", "sma_long",
        "ema_short", "ema_long",
        "rsi", "macd", "atr", "obv"
        ]
    ]
    return features

def build_return_features(df):

    rets = calculate_returns(df["price"])

    vola = rets.rolling(window=PARAMETER["vol_window"])
    vola = vola.std()
    
    df["return"] = rets
    df["return_vola"] = vola

    features = df[
        ["asset_id", "date", 
        "return", "return_vola"
        ]
    ]

    return features

def ingest_price_features(data):

    sql = """
            INSERT OR IGNORE INTO price_features
            (asset_id, date, price, sma_short, sma_long, 
            ema_short, ema_long, rsi, macd, atr, obv)

            VALUES (:asset_id, :date, :price, :sma_short, 
            :sma_long, :ema_short, :ema_long, :rsi, :macd, :atr, :obv)
        """
    write_sql(sql, data)

def ingest_return_features(data):

    sql = """
            INSERT OR IGNORE INTO return_features
            (asset_id, date, return, return_vola)

            VALUES (:asset_id, :date, :return, :return_vola)
        """
    write_sql(sql, data)

def run():
    asset_ids = AssetRepository().get_ids()
    price_list = []
    return_list = []
    for id in asset_ids:
        raw_prices = read_prices(id)
        raw_prices["price"] = raw_prices["adj_close"]
        price_features = build_price_features(raw_prices)
        return_features = build_return_features(raw_prices)

        price_list.append(price_features)
        return_list.append(return_features)

    df_price_features = pd.concat(price_list, ignore_index=True)
    df_return_features = pd.concat(return_list, ignore_index=True)

    ingest_price_features(df_price_features)
    ingest_return_features(df_return_features)


if __name__ == "__main__":
    run()