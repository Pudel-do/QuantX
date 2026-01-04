import pandas as pd
import numpy as np
import datetime as dt
from core.finance_adapter import FinanceAdapter
from core.asset_adapter import AssetRepository
from misc.utils import read_json, get_last_business_day
from misc.schema_utils import write_sql

PARAMETER = read_json("parameter.json")

def get_prices():
    start = PARAMETER["base_start"]
    ticker_list = AssetRepository().get_tickers()

    dfs = []
    for ticker in ticker_list:
        prices = FinanceAdapter(ticker).get_trade_data(start)
        prices["ticker"] = ticker
        prices.reset_index(names="date", inplace=True)
        dfs.append(prices)

    df_prices = pd.concat(dfs, ignore_index=True)
    return df_prices

def ingest_data(data):
    df = AssetRepository().get_ticker_id(data)
    df.columns = [col.replace(" ", "") for col in df.columns]
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    sql = """
            INSERT OR IGNORE INTO raw_prices 
            (asset_id, date, high, low, open, close, adj_close, volume)
            VALUES (:asset_id, :date, :High, :Low, :Open, :Close, :AdjClose, :Volume)
        """
    write_sql(sql, df)

def run():
    prices = get_prices()
    ingest_data(prices)

if __name__ == "__main__":
    run()