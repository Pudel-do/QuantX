import pandas as pd
import numpy as np
import yfinance as yf
import requests
import warnings
import logging
from core import logging_config
from alpha_vantage.fundamentaldata import FundamentalData
from datetime import datetime
from bs4 import BeautifulSoup
from misc.misc import *

class FinanceAdapter:
    def __init__(self, tick):
        self.tick = tick
        self.fx_config = read_json("exchange_rates.json")
        self.fd_config = read_json("constant.json")["fundamentals"]

    def get_quotes(self, start):
        """Function extracts all converted price data information and volume data for single ticker

        :param start: Start date for retrieving stock data for single ticker
        :type start: string
        :return: Dataframe with all price and volume data information. 
        :rtype: Dataframe
        """
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
        fd_list = self._establish_fmb_connection(fd_kpi=fd_kpi)
        fd_list_converted = self._fundamental_converter(fd_list=fd_list)
        fd = pd.DataFrame(fd_list_converted).set_index("date")
        return fd

    def _establish_fmb_connection(self, fd_kpi):
        fd_kpi_values = self.fd_config
        fd_kpi_value = fd_kpi_values[fd_kpi]
        api_key = os.getenv("API_KEY_FMP")
        url = f"https://financialmodelingprep.com/api/v3/{fd_kpi_value}/{self.tick}?period=annual&apikey={api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            fd_list = response.json()
            return fd_list
        else:
            logging.error(f"Connection to FMP failed for ticker {self.tick}")
            return []
        
    def _fundamental_converter(self, fd_list):
        fd_list_converted = []
        for item in fd_list:
            fx_ticker_mapping = self.fx_config["currency_ticker"]
            currency = item["reportedCurrency"]
            fd_start_date = get_business_day(item["date"])
            fd_end_date = pd.to_datetime(fd_start_date)
            fd_end_date = fd_end_date + pd.offsets.BDay(1)
            fd_end_date = fd_end_date.strftime(format="%Y-%m-%d")
            try:
                fx_ticker = fx_ticker_mapping[currency]
                try:
                    fx_rate = yf.download(
                        tickers=fx_ticker,
                        start=fd_start_date,
                        end=fd_end_date,
                        progress=False,
                        interval="1d",
                        ignore_tz=True,
                    )
                    converter = float(fx_rate["Close"].values)
                except KeyError:
                    logging.error(f"Exchange rate download failed for {fx_ticker}")
            except KeyError:
                logging.warning(f"Ticker {currency} not in config file")
                converter = 1
            fd_dict = {
                key: value * converter if isinstance(value, int) \
                    and not currency == self.fx_config["base_currency"] \
                    else value for key, value in item.items()
                    }
            fd_list_converted.append(fd_dict)
        return fd_list_converted


    def _quote_converter(self, data, currency):
        fx_ticker_mapping = self.fx_config["currency_ticker"]
        try:    
            fx_ticker = fx_ticker_mapping[currency]
            try:
                start = data.index[0]
                end = data.index[-1]
                fx_rate = yf.download(
                    tickers=fx_ticker,
                    start=start,
                    end=end,
                    progress=False,
                    interval="1d",
                    ignore_tz=True
                )["Close"]
                
                converter = pd.DataFrame(index=data.index)
                converter = converter.join(fx_rate)
                converter = converter.ffill()
                converter = np.array(converter).flatten()
            except KeyError:
                logging.error(f"Exchange rate download failed for {fx_ticker}")
        except KeyError:
            logging.warning(f"Ticker {currency} not in config file")
            converter = np.ones(shape=(len(data), ))

        gen = (col for col in data.columns if not col == "Volume")
        for col in gen:
            data[col] = data[col] * converter
        return data
    