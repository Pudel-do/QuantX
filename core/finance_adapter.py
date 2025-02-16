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

    def get_trade_data(self, start):
        """Function extracts all currency converted price data information 
        and volume data from Yahoo Finance for single ticker

        :param start: Start date for retrieving stock data for single ticker
        :type start: String
        :return: Dataframe with all price and volume data information. 
        :rtype: Dataframe
        """
        last_bday = get_last_business_day()
        end = pd.to_datetime(last_bday) + pd.offsets.BDay(1)
        quotes = yf.download(
            tickers=self.tick,
            start=start,
            end=end,
            progress=False,
            interval="1d",
            ignore_tz=True
        )
        if quotes.empty:
            logging.error(f"No quote data available for ticker {self.tick}")
        try:
            currency = yf.Ticker(self.tick).info["currency"]
        except KeyError:
            logging.error(f"Currency information error for ticker {self.tick}")
            return quotes
        base_currency = self.fx_config["base_currency"]
        if not currency == base_currency:
            quotes = self._quote_converter(quotes, currency)
        return quotes
    
    def get_last_quote(self):
        actual_date = datetime.now()
        start = get_business_day(date=actual_date)
        quotes = yf.download(
            tickers=self.tick,
            start=start,
            progress=False,
            interval="1d",
            ignore_tz=True
        )
        if quotes.empty:
            logging.error(f"No quote data available for ticker {self.tick}")
        elif len(quotes) > 1:
            logging.warning(f"Multiple quote data available for ticker {self.tick}")
            quotes = quotes.iloc[-1, :]
        else:
            pass
        try:
            currency = yf.Ticker(self.tick).info["currency"]
        except KeyError:
            logging.error(f"Currency information error for ticker {self.tick}")
            return quotes
        base_currency = self.fx_config["base_currency"]
        if not currency == base_currency:
            quotes = self._quote_converter(quotes, currency)
        return quotes
    
    def get_fundamental(self, fd_kpi):
        """Function extracts converted fundamental stock data from FMP

        :param fd_kpi: Considered fundamental KPI. 
        Can be <income>, <balance_sheet> or <cashflow>
        :type fd_kpi: String
        :return: Daframe contaiing fundamental stock data. 
        If no fundamental data was extracted, the dataframe is empty
        :rtype: Dataframe
        """
        fd_list = self._establish_fmb_connection(fd_kpi=fd_kpi)
        fd_list_converted = self._fundamental_converter(fd_list=fd_list)
        fd = pd.DataFrame(fd_list_converted)
        if not fd.empty:
            fd.set_index("date", inplace=True)
            fd.index = pd.to_datetime(fd.index).normalize()
        return fd
    
    def get_company_name(self):
        """Function returns company name for given tick

        :return: Company name
        :rtype: String
        """
        tick_info = yf.Ticker(self.tick).info
        name = tick_info["shortName"]
        return name

    def _establish_fmb_connection(self, fd_kpi):
        """Function connects to FMP via API call. If quarterly 
        fundamental data is needed, set period to <quarter>

        :param fd_kpi: String which fundamental data to download
        :type fd_kpi: String
        :return: Raw fundamental data for given stock
        :rtype: Json file
        """
        api_key = os.getenv("API_KEY_FMP")
        url = f"https://financialmodelingprep.com/api/v3/{fd_kpi}/{self.tick}?period=annual&apikey={api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            fd_list = response.json()
            return fd_list
        else:
            logging.warning(f"FMP conenction failed for ticker {self.tick}")
            return []
        
    def _fundamental_converter(self, fd_list):
        """Converts fundamental data values to defined base currency. 
        Fundamental data values for conversion refers to all integer data values. 
        All other fundamental data is adopted from raw value

        :param fd_list: List containing fundamental data for selected perido
        :type fd_list: List
        :return: Converted fundamental data to base currency
        :rtype: List
        """
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
        """Function converts stock quote data to base currency. 
        Volume columns remains unchanged.

        :param data: Minimum, maximum, open, close, adjusted close and volume stock data
        :type data: Dataframe
        :param currency: Currency from stock for fx ticker mapping
        :type currency: String
        :return: Converted stock quote data to base currency
        :rtype: Dataframe
        """
        fx_ticker_mapping = self.fx_config["currency_ticker"]
        try:    
            fx_ticker = fx_ticker_mapping[currency]
            try:
                if len(data) > 1:
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
                else:
                    start = data.index[0]
                    start = start.strftime(format="%Y-%m-%d")
                    fx_rate = yf.download(
                        tickers=fx_ticker,
                        start=start,
                        progress=False,
                        interval="1d",
                        ignore_tz=True
                    )["Close"] 
                    converter = fx_rate.iloc[0]
            except KeyError:
                logging.error(f"Exchange rate download failed for {fx_ticker}")
        except KeyError:
            logging.warning(f"Ticker {currency} not in config file")
            converter = np.ones(shape=(len(data), ))

        gen = (col for col in data.columns if not col == "Volume")
        for col in gen:
            data[col] = data[col] * converter
        return data
    