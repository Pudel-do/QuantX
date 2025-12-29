import pandas as pd
import numpy as np
from sqlalchemy import text
from db.engine import get_engine
from core.finance_adapter import FinanceAdapter
from misc.utils import *

PARAMETER = read_json("parameter.json")


def run():
    rows = []
    for ticker in PARAMETER["ticker"]:
        info = FinanceAdapter(ticker).get_stock_infos()
        rows.append({
            "ticker": ticker,
            "name_long": info["longName"],
            "name_short": info["shortName"],
            "currency": info["currency"]
        })
    df = pd.DataFrame(rows)

    engine = get_engine()
    with engine.begin() as conn:
        conn.execute(
            text("""
            INSERT OR IGNORE INTO tickers (TICKER, NAME_LONG, NAME_SHORT, CURRENCY)
            VALUES (:ticker, :name_long, :name_short, :currency)
            """),
            df.to_dict(orient="records")
        )

if __name__ == "__main__":
    run()