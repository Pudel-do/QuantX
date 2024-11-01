import pandas as pd
import pickle
import logging
import keras
import os
from misc.misc import read_json


class FileAdapter:
    def __init__(self) -> None:
        self.config = read_json("constant.json")["datamodel"]
    
    def save_dataframe(self, df, path, file_name):
        """Function saves dataframe to to given
        directory and file name

        :param df: Dataframe to save
        :type df: Dataframe
        :return: None
        :rtype: None
        """
        dir = os.path.join(
            os.getcwd(), 
            path
        )
        self._write_csv(
            data=df,
            dir=dir,
            file_name=file_name
        )
        return None
    
    def save_object(self, obj, path, file_name):
        """Function saves object to given directory 
        and file name as pickel file

        :param obj: Object to save
        :type obj: Dictionary or class
        :return: None
        :rtype: None
        """
        dir = os.path.join(
            os.getcwd(), 
            path
        )
        self._write_pickel(
            data=obj,
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
        file_name = model.model_id
        self._write_pickel(
            data=model,
            dir=dir,
            file_name=file_name
        )
        return None
    
    def load_dataframe(self, path, file_name):
        """Function loads closing quotes from data directory

        :return: Closing quotes
        :rtype: Dataframe
        """
        dir = os.path.join(
            os.getcwd(), 
            path
        )
        df = self._load_csv(
            dir=dir, 
            file_name=file_name
            )
        return df
    
    def load_object(self, path, file_name):
        """Function loads fundamental stock data from data directory

        :return: Fundamental data
        :rtype: Dictionary
        """
        dir = os.path.join(
            os.getcwd(), 
            path
        )
        obj = self._load_pickel(
            dir=dir,
            file_name=file_name
        )
        return obj
    
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
        path = os.path.join(dir, file_name) + ".csv"
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
        path = os.path.join(dir, file_name) + ".pkl"
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
        path = os.path.join(dir, file_name) + ".csv"
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
        path = os.path.join(dir, file_name) + ".pkl"
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
