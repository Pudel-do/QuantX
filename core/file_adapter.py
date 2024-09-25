import pandas as pd
import pickle
import logging
import keras
import os
from misc.misc import read_json


class FileAdapter:
    def __init__(self) -> None:
        self.config = read_json("constant.json")["datamodel"]

    def save_stock_returns(self, rets):
        """Function saves stock return data for
        different stocks to data directory as csv file

        :param rets: Stock returns
        :type data: Dataframe
        :return: None
        :rtype: None
        """
        dir = os.path.join(
            os.getcwd(), 
            self.config["data_dir"]
        )
        file_name = self.config["returns_file"]
        self._write_csv(
            data=rets,
            dir=dir,
            file_name=file_name
        )
        return None
    
    def save_closing_quotes(self, quotes):
        """Function saves closing quotes for different
        stocks to data directory as csv file

        :param quotes: Closing quotes
        :type quotes: Dataframe
        :return: None
        :rtype: None
        """
        dir = os.path.join(
            os.getcwd(), 
            self.config["data_dir"]
        )
        file_name = self.config["quotes_file"]
        self._write_csv(
            data=quotes,
            dir=dir,
            file_name=file_name
        )
        return None
    
    def save_fundamentals(self, fundamentals):
        """Function saves fundamental data for
        single stock to data directory as pickle file

        :param fundamentals: Fundamental data
        :type funds: Dictionary
        :return: None
        :rtype: None
        """
        dir = os.path.join(
            os.getcwd(), 
            self.config["data_dir"]
        )
        file_name = self.config["fundamentals_file"]
        self._write_pickel(
            data=fundamentals,
            dir=dir,
            file_name=file_name
        )
        return None
    
    def save_trading_data(self, trading_data):
        """Function saves daily trading data for single
        stock to feature directory as pickel file

        :param trading_data: Daily trading data
        :type trading_data: Dictionary
        :return: None
        :rtype: None
        """
        dir = os.path.join(
            os.getcwd(), 
            self.config["feature_dir"]
        )
        file_name = self.config["daily_trading_data_file"]
        self._write_pickel(
            data=trading_data,
            dir=dir,
            file_name=file_name
        )
        return None
    
    def save_model_data(self, model_data):
        dir = os.path.join(
            os.getcwd(),
            self.config["data_dir"],
        )
        file_name = self.config["model_data"]
        self._write_pickel(
            data=model_data,
            dir=dir,
            file_name=file_name
        )
        return None

    def save_model(self, model):
        dir = os.path.join(
            os.getcwd(), 
            self.config["models_dir"],
            model.ticker,
            self.config["model"]
        )
        file_name = f"{model.model_id}.pkl"
        self._write_pickel(
            data=model,
            dir=dir,
            file_name=file_name
        )
        return None
    
    def load_closing_quotes(self):
        """Function loads closing quotes from data directory

        :return: Closing quotes
        :rtype: Dataframe
        """
        dir = os.path.join(
            os.getcwd(), 
            self.config["data_dir"]
        )
        file_name = self.config["quotes_file"]
        closing_quotes = self._load_csv(
            dir=dir, 
            file_name=file_name
            )
        return closing_quotes
    
    def load_returns(self):
        """Function loads stock returns from data directory

        :return: Stock returns
        :rtype: Dataframe
        """
        dir = os.path.join(
            os.getcwd(), 
            self.config["data_dir"]
        )
        file_name = self.config["returns_file"]
        returns = self._load_csv(
            dir=dir,
            file_name=file_name
        )
        return returns
    
    def load_fundamentals(self):
        """Function loads fundamental stock data from data directory

        :return: Fundamental data
        :rtype: Dictionary
        """
        dir = os.path.join(
            os.getcwd(), 
            self.config["data_dir"]
        )
        file_name = self.config["fundamentals_file"]
        fundamentals = self._load_pickel(
            dir=dir,
            file_name=file_name
        )
        return fundamentals
    
    def load_trading_data(self):
        """Function loads daily trading stock data from feature directory

        :return: Daily trading data
        :rtype: Dictionary
        """
        dir = os.path.join(
            os.getcwd(), 
            self.config["feature_dir"]
        )
        file_name = self.config["daily_trading_data_file"]
        trading_data = self._load_pickel(
            dir=dir, 
            file_name=file_name
        )
        return trading_data
    
    def load_model(self, ticker, model_id):
        dir = os.path.join(
            os.getcwd(),
            self.config["models_dir"],
            ticker,
            self.config["model"]
        )
        model = self._load_pickel(
            dir=dir,
            file_name=model_id
        )
        return model

    def _write_csv(self, data, dir, file_name):
        """Function writes data to given directory 
        as csv file

        :param data: Data to write to working directory
        :type data: Dataframe
        :param dir: Directory where data should be written to
        :type dir: String
        :param file_name: File name for saving data
        :type file_name: String
        :return: None
        :rtype: None
        """
        self._dir_check(dir=dir)
        path = os.path.join(dir, file_name)
        data.to_csv(path,
                    sep=",",
                    decimal=".",
                    index=True
                    )
        return None
    
    def _write_pickel(self, data, dir, file_name):
        """Function writes data to given directory
        as pickel file


        :param data: Data to write to working directory
        :type data: Dataframe
        :param dir: Directory where data should be written to
        :type dir: String
        :param file_name: File name for saving data
        :type file_name: String
        :return: None
        :rtype: None
        """
        self._dir_check(dir=dir)
        path = os.path.join(dir, file_name)
        with open(path, "wb") as file:
            pickle.dump(data, file=file)
        return None
    
    def _load_csv(self, dir, file_name):
        """Function loads csv file from given
        directory and file_name

        :param dir: Directory to load data from
        :type dir: String
        :param file_name: Csv file to load
        :type file_name: String
        :return: Data from csv file
        :rtype: Dataframe
        """
        path = os.path.join(dir, file_name)
        try:
            data = pd.read_csv(
                path,
                sep=",",
                decimal=".",
                index_col=0,
                parse_dates=True
            )
            return data
        except FileNotFoundError:
            logging.error(f"File {file_name} from {dir} directory not found")
            return None
        return data
    
    def _load_pickel(self, dir, file_name):
        """Function loads pickel file from given
        directory and file_name

        :param dir: Directory to load data from
        :type dir: String
        :param file_name: Pickel file to load
        :type file_name: Strong
        :return: Data from pickel file
        :rtype: Pickel object
        """
        path = os.path.join(dir, file_name)
        try:
            with open(path, "rb") as file:
                data = pickle.load(file=file)
            return data
        except:
            logging.error(f"File {file_name} from {dir} directory not found")
            return None

    def _dir_check(self, dir):
        """Function checks if given directory exists.
        If directory not exists, the function creates directory

        :param dir: Directory to check for
        :type dir: String
        :return: None
        :rtype: None
        """
        if not os.path.exists(dir):
            os.mkdir(dir) 
        else:
            pass
        return None
