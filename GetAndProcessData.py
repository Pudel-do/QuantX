import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from core.finance_adapter import FinanceAdapter
from misc.misc import read_json
from itertools import chain

def concat_quotes(ticker_list, start, quote_id):
    end = datetime.now()
    quotes = pd.DataFrame(
        index=pd.date_range(
            start=start,
            end=end,
            freq="B",
            normalize=True
        )
    )

    for tick in ticker_list:
        ticker_quotes = FinanceAdapter(tick).get_quotes(start=start)
        ticker_quote = ticker_quotes[quote_id]
        ticker_quote.name = tick
        quotes = quotes.join(ticker_quote)
    return quotes

def calculate_returns(quotes):
    rets = np.log(quotes / quotes.shift(1))
    rets = rets.iloc[1:]
    return rets
    
def preprocess_fundamentals(ticker_list, start):
    fundamental_dict = {}
    for tick in ticker_list:
        income_statement = FinanceAdapter(tick).get_fundamental(fd_kpi="income")
        balance_sheet = FinanceAdapter(tick).get_fundamental(fd_kpi="balance_sheet")
        cashflow = FinanceAdapter(tick).get_fundamental(fd_kpi="cashflow")
        fundamentals = pd.concat(
            objs=[
                income_statement,
                balance_sheet,
                cashflow
            ],
            axis=1
        )
        fundamental_dict[tick] = fundamentals
    return fundamental_dict

if __name__ == "__main__":
    ticker_list = read_json("inputs.json")["ticker"]
    base_start = read_json("inputs.json")["base_start"]
    closing_quotes = concat_quotes(ticker_list=ticker_list, 
                                   start=base_start, 
                                   quote_id="Adj Close")
    returns = calculate_returns(closing_quotes)
    fundamental_data = preprocess_fundamentals(ticker_list, base_start)


