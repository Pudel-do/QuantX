import pandas as pd
import numpy as np
import datetime as dt
from core.finance_adapter import FinanceAdapter
from misc.utils import read_json, get_last_business_day
from misc.schema_utils import get_ticker_id, write_sql

PARAMETER = read_json("parameter.json")

def get_prices():
    start = PARAMETER["base_start"]
    dfs = []
    for ticker in PARAMETER["ticker"]:
        prices = FinanceAdapter(ticker).get_trade_data(start)
        prices["ticker"] = ticker
        prices.reset_index(names="date", inplace=True)
        dfs.append(prices)

    df_prices = pd.concat(dfs, ignore_index=True)
    return df_prices

def ingest_raw_data(data):
    df = get_ticker_id(data)

    df.columns = [col.replace(" ", "") for col in df.columns]
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    sql = """
            INSERT OR IGNORE INTO raw_prices 
            (ticker_id, date, high, low, open, close, adj_close, volume)
            VALUES (:ticker_id, :date, :High, :Low, :Open, :Close, :AdjClose, :Volume)
        """
    write_sql(sql, df)

def run():
    prices = get_prices()
    ingest_raw_data(prices)

if __name__ == "__main__":
    run()