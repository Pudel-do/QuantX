import pandas as pd
import numpy as np
import datetime as dt
from core.finance_adapter import FinanceAdapter
from misc.utils import read_json, get_last_business_day
from misc.schema_utils import *


def read_prices(ticker_id):

    sql = """
        SELECT ticker_id, date, adj_close, volume
        FROM raw_prices
        WHERE ticker_id = :ticker_id
        ORDER BY date
        """
    
    df = read_sql(
        sql=sql,
        params={
            "ticker_id": ticker_id
        }
    )
    
    return df

def calculate_features():
    pass

def run():
    tickers = read_sql(
        sql= """
            SELECT ticker_id, ticker
            FROM tickers
            """,
        params=None
    )
    for _, row in tickers.iterrows():
        prices = read_prices(row["ticker_id"])

if __name__ == "__main__":
    run()