import pandas as pd
import numpy as np
import yfinance as yf
import requests
import warnings
import logging
from core import logging_config
from financetoolkit import Toolkit
from alpha_vantage.fundamentaldata import FundamentalData
from datetime import datetime
from bs4 import BeautifulSoup
from misc.misc import *

class FinanceAdapter:
    def __init__(self, tick):
        self.tick = tick
        self.fx_config = read_json("exchange_rates.json")
        self.fd_config = read_json("constant.json")["fundamentals"]
        pass

    def get_quotes(self, start):
        end = datetime.now()
        quotes = yf.download(
            tickers=self.tick,
            start=start,
            end=end,
            progress=False,
            interval="1d",
            ignore_tz=True
        )
        currency = yf.Ticker(self.tick).info["currency"]
        base_currency = self.fx_config["base_currency"]
        if not currency == base_currency:
            quotes = self._quote_converter(quotes, currency)

        return quotes
    
    def get_fundamental(self, fd_kpi):
        fd = self._establish_fmb_connection(fd_kpi=fd_kpi)
        return fd

    def _establish_fmb_connection(self, fd_kpi):
        fd_kpi_values = self.fd_config
        fd_kpi_value = fd_kpi_values[fd_kpi]
        api_key = os.getenv("API_KEY_FMP")
        url = f"https://financialmodelingprep.com/api/v3/{fd_kpi_value}/{self.tick}?period=annual&apikey={api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            fd_data = response.json()
            return fd_data
        else:
            logging.error(f"Connection to FMP failed for ticker {self.tick}")
            return []

    def _quote_converter(self, data, currency):
        try:
            exchange_rate_ticker = self.fx_config["currency_ticker"]
            ticker = exchange_rate_ticker[currency]
            try:
                start = data.index[0]
                end = data.index[-1]
                exchange_rates = yf.download(
                    tickers=ticker,
                    start=start,
                    end=end,
                    progress=False,
                    interval="1d",
                    ignore_tz=True
                )["Close"]
                
                converter = pd.DataFrame(index=data.index)
                converter = converter.join(exchange_rates)
                converter = converter.ffill()
                converter = np.array(converter).flatten()
            except KeyError:
                logging.error(f"Download failed for exchange rate symbol {ticker}")
        except KeyError:
            logging.warning(f"Ticker {currency} not in config file")
            converter = np.ones(shape=(len(data), ))

        gen = (col for col in data.columns if not col == "Volume")
        for col in gen:
            data[col] = data[col] * converter
            
        return data
    
    def _fundamental_converter(self):
        pass