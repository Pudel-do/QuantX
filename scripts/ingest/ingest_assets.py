import pandas as pd
import numpy as np
from sqlalchemy import text
from db.engine import get_engine
from core.finance_adapter import FinanceAdapter
from core.asset_adapter import AssetRepository
from misc.utils import read_json

PARAMETER = read_json("parameter.json")


def run():
    ticker_list = PARAMETER["ticker"]
    ticker_list.append(PARAMETER["benchmark_tick"])
    for ticker in ticker_list:
        AssetRepository().add_asset(ticker)
    AssetRepository().update_active_flag(ticker_list)

if __name__ == "__main__":
    run()