import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from core.finance_adapter import FinanceAdapter
from misc.misc import read_json

def combine_quotes(ticker, start, quote_id):
    end = datetime.now()
    quotes = pd.DataFrame(
        index=pd.date_range(
            start=start,
            end=end,
            freq="B",
            normalize=True
        )
    )

    for tick in ticker:
        ticker_quotes = FinanceAdapter().get_quotes(tick=tick, start=start)
        ticker_quote = ticker_quotes[quote_id]
        ticker_quote.name = tick
        quotes = quotes.join(ticker_quote)
    return quotes
    
def combine_fundamentals(ticker, start):
    liste = []
    for tick in ticker:
        balance_sheet = FinanceAdapter().get_balance_sheet(tick=tick)
        income_statement = FinanceAdapter().get_income_statement(tick=tick)
        print("break")

    return liste
    

if __name__ == "__main__":
    ticker = read_json("inputs.json")["ticker"]
    base_start = read_json("inputs.json")["base_start"]
    #closing_quotes = combine_quotes(ticker=ticker, start=base_start, quote_id="Adj Close")
    fundamental = combine_fundamentals(ticker, base_start)


