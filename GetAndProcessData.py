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

def get_merged_quotes(ticker_list, start, quote_id):
    """Function concats quotes for given quote id and ticker symbols. The ticker quotes are
    joined to a base time series with business days only ranging up to actual time.

    :param ticker_list: Tickers for merging stock quotes
    :type ticker_list: List
    :param start: Start date for downloading quote data
    :type start: String
    :param quote_id: Quote measure which can be "Low", "High", "Open", "Close" or "Adj Close"
    :type quote_id: String
    :return: Dataframe containig merged quotes for all ticker in ticker_list
    :rtype: Dataframe
    """
    end = get_last_business_day()
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
        ticker_quote = ticker_quotes[quote_id]
        ticker_quote.name = tick
        quotes = quotes.join(ticker_quote)
    return quotes

def get_daily_stock_data(ticker_list, start):
    """Function extracts daily tading data for opening, 
    closed, high, low, adjusted closed quotes and 
    trading volume for given ticker list and start date

    :param ticker_list: Ticker symbols for data extraction
    :type ticker_list: List
    :param start: Start date for data extraction
    :type start: String
    :return: Daily trading data
    :rtype: Dictionary
    """
    ticker_dict = {}
    for tick in ticker_list:
        data = FinanceAdapter(tick).get_trade_data(start=start)
        ticker_dict[tick] = data

    return ticker_dict

def get_returns(quotes):
    """Function calculates log returns for
    given quotes and time range

    :param quotes: Quotes from stocks
    :type quotes: Dataframe
    :return: Stock returns
    :rtype: Dataframe
    """
    rets = np.log(quotes / quotes.shift(1))
    rets = rets.iloc[1:]
    return rets
    
def get_fundamentals(ticker_list, start):
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
    ticker_list = read_json("inputs.json")["ticker"]
    base_start = read_json("inputs.json")["base_start"]
    CONST_FUNDS = read_json("constant.json")["fundamentals"]
    closing_quotes = get_merged_quotes(ticker_list=ticker_list, start=base_start, quote_id="Adj Close")
    returns = get_returns(closing_quotes)
    daily_trading_data = get_daily_stock_data(ticker_list, base_start)
    fundamentals = get_fundamentals(ticker_list, base_start)
    fundamentals = drop_duplicate_fundamental_cols(fundamentals)
    FileAdapter().save_closing_quotes(closing_quotes)
    FileAdapter().save_stock_returns(returns)
    FileAdapter().save_trading_data(daily_trading_data)
    FileAdapter().save_fundamentals(fundamentals)



