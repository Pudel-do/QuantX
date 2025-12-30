import pandas as pd
import numpy as np
import datetime as dt
from sqlalchemy import text
from db.engine import get_engine
from core.finance_adapter import FinanceAdapter
from misc.utils import read_json, get_last_business_day
from misc.schema_utils import get_ticker_id

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
    engine = get_engine()
    df = get_ticker_id(
        df=data,
        engine=engine
    )

    df.columns = [col.replace(" ", "") for col in df.columns]
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    with engine.begin() as conn:
        conn.execute(
            text("""
            INSERT OR IGNORE INTO raw_prices (ticker_id, date, high, low, open, close, adj_close, volume)
            VALUES (:ticker_id, :date, :High, :Low, :Open, :Close, :AdjClose, :Volume)
            """),
            df.to_dict(orient="records")
        )

def run():
    prices = get_prices()
    ingest_raw_data(prices)

if __name__ == "__main__":
    run()