import pandas as pd
import os
from misc.misc import read_json

class FileAdapter:
    def __init__(self) -> None:
        self.config = read_json("constant.json")["datamodel"]

    def write_stock_rets(self, rets):
        """Function writes stock return data
        to data directory as csv file

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

    def _write_csv(self, data, dir, file_name):
        """Function writes data to working
        directory as csv file

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
