import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from core.finance_adapter import FinanceAdapter
from core.file_adapter import FileAdapter
from itertools import chain
from misc.misc import *

def drop_duplicate_fundamental_cols(fundamentals):
    """Function drops duplicate fundamental KPIs

    :param fundamentals: Dictionary for fundamental data
    with ticker symbols as keys
    :type fundamentals: Dictionary
    :return: Fundamental dictionary adjusted by duplicate
    fundamental KPIs
    :rtype: Dictionary
    """
    fundamentals_clean = {}
    for tick, funds in fundamentals.items():
        dup_mask = funds.columns.duplicated()
        funds = funds.loc[:, ~dup_mask]
        fundamentals_clean[tick] = funds

    return fundamentals_clean

def get_merged_quotes(ticker_list, quote_id):
    """Function concats quotes for given quote id and ticker symbols. The ticker quotes are
    joined to a base time series with business days only ranging up to actual time.

    :param ticker_list: Tickers for merging stock quotes
    :type ticker_list: List
    :param quote_id: Quote measure which can be "Low", "High", "Open", "Close" or "Adj Close"
    :type quote_id: String
    :return: Dataframe containig merged quotes for all ticker in ticker_list
    :rtype: Dataframe
    """
    if PARAMETER["base_end"] is None:
        end = get_last_business_day()
    else:
        end = PARAMETER["base_end"]
    start = PARAMETER["base_start"]
    quotes = pd.DataFrame(
        index=pd.date_range(
            start=start,
            end=end,
            freq="B",
            normalize=True
        )
    )

    for tick in ticker_list:
        ticker_quotes = FinanceAdapter(tick).get_trade_data(start=start)
        ticker_quotes = rename_yfcolumns(data=ticker_quotes)
        ticker_quote = ticker_quotes[quote_id]
        if not ticker_quote.empty:
            ticker_quote.name = tick
            quotes = quotes.join(ticker_quote, how="left")
    return quotes

def get_daily_stock_data(ticker_list):
    """Function extracts daily tading data for opening, 
    closed, high, low, adjusted closed quotes and 
    trading volume for given ticker list and start date
    and adjusts column names to constant columns

    :param ticker_list: Ticker symbols for data extraction
    :type ticker_list: List
    :return: Daily trading data
    :rtype: Dictionary
    """
    ticker_dict = {}
    start = PARAMETER["base_start"]
    for tick in ticker_list:
        data = FinanceAdapter(tick).get_trade_data(start=start)
        data = rename_yfcolumns(data=data)
        ticker_dict[tick] = data

    return ticker_dict

def get_stock_infos(ticker_list):
    df_infos = pd.DataFrame()
    info_list = PARAMETER["stock_infos"]
    for tick in ticker_list:
        try:
            stock_infos = FinanceAdapter(tick).get_stock_infos()
            for info in info_list:
                df_infos.loc[info, tick] = stock_infos[info]
        except KeyError:
            logging.warning(f"No information available for {tick}")
            df_infos = pd.DataFrame(columns=ticker_list)
    return df_infos

    
def get_fundamentals(ticker_list):
    """Function extracts quarterly currency converted fundamental data 
    from FMP for given ticker symbols and start date up to current date.
    Fundamental data is filtered for columns defined in config file for 
    constant definitions. Data is excluded from the function output 
    if no fundamental data could be extracted. 

    :param ticker_list: Ticker symbols for extracting data
    :type ticker_list: List
    :param start: Start date for extracting data
    :type start: String
    :return: Fundamental data
    :rtype: Dictionary
    """
    fundamental_dict = {}
    for tick in ticker_list:
        fd_adapter = FinanceAdapter(tick)
        income_statement = fd_adapter.get_fundamental(
            fd_kpi=CONST_FUNDS["income"]
            )
        balance_sheet = fd_adapter.get_fundamental(
            fd_kpi=CONST_FUNDS["balance_sheet"]
            )
        cashflow = fd_adapter.get_fundamental(
            fd_kpi=CONST_FUNDS["cashflow"]
            )
        fundamentals = pd.concat(
            objs=[
                income_statement,
                balance_sheet,
                cashflow
            ],
            axis=1
        )
        if not fundamentals.empty:
            fundamentals = fundamentals[CONST_FUNDS["measures"]]
            fundamental_dict[tick] = fundamentals
        else:
            pass

    return fundamental_dict

if __name__ == "__main__":
    ticker_list = read_json("parameter.json")["ticker"]
    PARAMETER = read_json("parameter.json")
    CONST_FUNDS = read_json("constant.json")["fundamentals"]
    CONST_COLS = read_json("constant.json")["columns"]
    CONST_DATA = read_json("constant.json")["datamodel"]
    stock_ticks = PARAMETER["ticker"]
    benchmark_tick = [PARAMETER["benchmark_tick"]]
    stock_quotes = get_merged_quotes(
        ticker_list=ticker_list, 
        quote_id=PARAMETER["quote_id"]
    )
    benchmark_quotes = get_merged_quotes(
        ticker_list=benchmark_tick, 
        quote_id=PARAMETER["quote_id"]
    )
    stock_returns = calculate_returns(stock_quotes)
    benchmark_returns = calculate_returns(benchmark_quotes)
    daily_trading_data = get_daily_stock_data(ticker_list)
    fundamentals = get_fundamentals(ticker_list)
    fundamentals = drop_duplicate_fundamental_cols(fundamentals)
    stock_infos = get_stock_infos(ticker_list)
    FileAdapter().save_dataframe(
        df=stock_quotes,
        path=CONST_DATA["raw_data_dir"],
        file_name=CONST_DATA["quotes_file"]
    )
    FileAdapter().save_dataframe(
    df=stock_returns,
    path=CONST_DATA["raw_data_dir"],
    file_name=CONST_DATA["stock_returns_file"]
    )
    FileAdapter().save_dataframe(
    df=benchmark_returns,
    path=CONST_DATA["raw_data_dir"],
    file_name=CONST_DATA["benchmark_returns_file"]
    )
    FileAdapter().save_dataframe(
        df=stock_infos,
        path=CONST_DATA["raw_data_dir"],
        file_name=CONST_DATA["stock_infos"]
    )
    FileAdapter().save_object(
        obj=daily_trading_data,
        path=CONST_DATA["feature_dir"],
        file_name=CONST_DATA["daily_trading_data_file"]
    )
    FileAdapter().save_object(
        obj=fundamentals,
        path=CONST_DATA["raw_data_dir"],
        file_name=CONST_DATA["fundamentals_file"]
    )



