import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from core.yfinance_adapter import YahooFinance
from misc.misc import *

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
        quotes = YahooFinance().get_quotes(tick=tick, start=start)
        print("break")
    
    

if __name__ == "__main__":
    ticker = read_json("inputs.json")["ticker"]
    base_start = read_json("inputs.json")["base_start"]
    closing_quotes = combine_quotes(ticker=ticker, start=base_start, quote_id="Adj Close")
    print("break")

