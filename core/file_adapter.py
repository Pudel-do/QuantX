import pandas as pd
import pickle
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
        dir = os.path.join(os.getcwd(), self.config["data_dir"])
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
        dir = os.path.join(os.getcwd(), self.config["data_dir"])
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
        dir = os.path.join(os.getcwd(), self.config["data_dir"])
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
        dir = os.path.join(os.getcwd(), self.config["feature_dir"])
        file_name = self.config["daily_trading_data_file"]
        self._write_pickel(
            data=trading_data,
            dir=dir,
            file_name=file_name
        )
        return None

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
