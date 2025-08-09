import pandas as pd
import numpy as np
import logging
import math
import warnings
from misc.utils import *
from core.portfolio_generator import PortfolioGenerator
from core.file_adapter import FileAdapter
from core.finance_adapter import FinanceAdapter
warnings.filterwarnings('ignore')

def return_cleaning(returns):
    """Function preprocesses return data
    for clean data set as base for portfolio
    construction. Preprocessing includes
    portfolio period filtering and fill nan
    values with zero

    :param returns: Stock returns
    :type returns: Dataframe
    :return: Clean returns data
    :rtype: Dataframe
    """
    returns_clean = returns.iloc[1:]
    return returns_clean

if __name__ == "__main__":
    CONST_COLS = read_json("constant.json")["columns"]
    CONST_KEYS = read_json("constant.json")["keys"]
    CONST_DATA = read_json("constant.json")["datamodel"]
    PARAMETER = read_json("parameter.json")
    raw_stock_returns = FileAdapter().load_dataframe(
        path=CONST_DATA["raw_data_dir"],
        file_name=CONST_DATA["stock_returns_file"]
    )
    raw_stock_returns, tickers = harmonize_tickers(
        object=raw_stock_returns
    )
    raw_bench_returns = FileAdapter().load_dataframe(
        path=CONST_DATA["raw_data_dir"],
        file_name=CONST_DATA["benchmark_returns_file"]
    )

    stock_returns = return_cleaning(raw_stock_returns)
    bench_returns = return_cleaning(raw_bench_returns)
    future_stock_returns = get_future_returns(
        tickers=PARAMETER["ticker"],
        rets=stock_returns
    )

    FileAdapter().save_dataframe(
        df=stock_returns,
        path=CONST_DATA["processed_data_dir"],
        file_name=CONST_DATA["stock_returns_file"]
    )
    FileAdapter().save_dataframe(
        df=bench_returns,
        path=CONST_DATA["processed_data_dir"],
        file_name=CONST_DATA["benchmark_returns_file"]
    )
    FileAdapter().save_dataframe(
        df=future_stock_returns,
        path=CONST_DATA["processed_data_dir"],
        file_name=CONST_DATA["future_stock_returns_file"]
    )
    
