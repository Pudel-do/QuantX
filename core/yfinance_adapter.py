import pandas as pd
import numpy as np
import yfinance as yf
import requests
import warnings
from datetime import datetime
from config import Config
from bs4 import BeautifulSoup

warnings.filterwarnings("ignore", category=FutureWarning)

class YahooFinance:
    def __init__(self):
        self.date_format = '%Y-%m-%d'
        self.quote_id = 'Adj Close'

    def _exchange_rates_conversion(self, quotes, currency_id):
        """
        :param quotes:
        :param currency_id:
        :return: Series containing converted quotes for given asset
        """
        currency_mapping = Config('parameter.json').config['ExchangeRates']
        if not currency_id == 'EUR':
            currency_tick = currency_mapping[currency_id]
            try:
                start = quotes.index[0].strftime(self.date_format)
                end = quotes.index[-1].strftime(self.date_format)
                exchange_rates = yf.download(currency_tick,
                                             start,
                                             end,
                                             progress=False,
                                             interval='1d',
                                             ignore_tz=True)
                exchange_rates = exchange_rates[self.quote_id]
                if not exchange_rates.empty:
                    exchange_rates.fillna(method='ffill', inplace=True)
                    converted_quotes = quotes.mul(exchange_rates)
                    converted_quotes = converted_quotes.dropna()
            except:
                print(f'Warning: Currency {currency_id} not available for quote conversion')
                converted_quotes = quotes
        else:
            converted_quotes = quotes

        return converted_quotes

    def get_quotes(self, ticks, start, end):
        """
        :param start: String for start date to get beginning of quotes
        :param end: String for end date to get end of quotes
        :param ticks: List of ticker symbols for quotes
        :return: Dataframe containing converted quotes
        """
        quotes_index = pd.date_range(start,
                                     end,
                                     freq='B',
                                     normalize=True)
        quotes = pd.DataFrame(index=quotes_index)
        for tick in ticks:
            print(f"--- Downloading quotes for ticker {tick} ---")
            try:
                tick_quotes = yf.download(tick,
                                          start,
                                          end,
                                          progress=False,
                                          interval='1d',
                                          ignore_tz=True)
                tick_currency = yf.Ticker(tick).fast_info['currency']
            except:
                print(f"Warning: Downloading quotes for ticker {tick} failed")

            tick_quotes = tick_quotes[self.quote_id]
            tick_quotes_adj = self._exchange_rates_conversion(tick_quotes, tick_currency)
            tick_quotes_adj.name = tick
            quotes = quotes.join(tick_quotes_adj)
        return quotes

    def get_last_quote(self, ticks):
        """
        :param ticks: List of ticker symbols for quotes
        :return: Dataframe with ticks as columns containing the last valid quote
        """

        end = pd.Timestamp.today()
        start = end - pd.DateOffset(months=1)
        end = end.strftime(self.date_format)
        start = start.strftime(self.date_format)
        last_quotes = []
        for tick in ticks:
            print(f"--- Downloading last valid quote for ticker {tick} ---")
            try:
                tick_quotes = yf.download(tick,
                                        start,
                                        end,
                                        progress=False,
                                        interval='1d',
                                        ignore_tz=True)
            except:
                print(f"Warning: Downloading quote for ticker {tick} failed")

            tick_currency = yf.Ticker(tick).fast_info['currency']
            tick_quotes = tick_quotes[self.quote_id]
            tick_quotes = self._exchange_rates_conversion(tick_quotes, tick_currency)
            index = tick_quotes.last_valid_index()
            quote = tick_quotes.loc[index]
            quote = np.round(quote, 3)
            last_quotes.append(quote)

        last_quotes = np.array(last_quotes)
        return last_quotes

    def get_returns(self, ticks, start, end):
        """
        :param start: String with start date for return calculation
        :param end: String with end date for return calculation
        :param ticks: List of ticker symbols for return calculation
        :return: Dataframe containing returns for given ticker symbols and start and end date
        """
        quotes = self.get_quotes(ticks, start, end)
        rets = np.log(quotes / quotes.shift(1))
        rets = rets.iloc[1:]
        return rets

    def get_tick_name(self, ticker):
        """
        :param ticker: String for ticker symbol
        :return: String with company long name for given ticker if web scrapping is succesfull. Otherwise return none.
        """
        yfinance = "https://query2.finance.yahoo.com/v1/finance/search"
        user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36'
        params = {"q": ticker, "quotes_count": 1}
        request = requests.get(url=yfinance, params=params, headers={'User-Agent': user_agent})
        if not request.status_code == 404:
            try:
                tick_data = request.json()
                tick_name = tick_data['quotes'][0]['longname']
                return tick_name
            except:
                print(f'Warning: Downloading name for ticker {ticker} not possible')
                return None
        else:
            print('Warning: URL for scrapping company name by ticker not valid anymore')
            return None

    def get_tick_news(self, ticks):
        for tick in ticks:
            articles = yf.Ticker(tick).news
            for article in articles:
                timestamp_int = article['providerPublishTime']
                timestamp = datetime.fromtimestamp(timestamp_int)
                article_url = article['link']
                page = requests.get(url=article_url)
                soup = BeautifulSoup(page.content, "html.parser")
                #ToDO: Parse html file to text
        return
