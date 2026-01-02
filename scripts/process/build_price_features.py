import pandas as pd
import numpy as np
import datetime as dt
from core.finance_adapter import FinanceAdapter
from core.asset_adapter import AssetRepository
from misc.utils import read_json, get_last_business_day
from misc.schema_utils import *


def read_prices(asset_id):

    sql = """
        SELECT asset_id, date, adj_close, volume
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

def calculate_features():
    pass

def run():
    asset_ids = AssetRepository().get_ids()
    for id in asset_ids:
        prices = read_prices(id)
        print("break")

if __name__ == "__main__":
    run()