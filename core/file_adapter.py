import pandas as pd
import os
from misc.misc import read_json

class FileAdapter:
    def __init__(self) -> None:
        self.config = read_json("constant.json")["datamodel"]

    def write_stock_returns(self, data):
        dir = os.path.join(os.getcwd(), self.config["data_path"])
        file_name = self.config["returns_file"]

        pass

    def _write_csv(self, data, dir, file_name):
        self._dir_check(dir=dir)
        path = os.path.join(dir, file_name)
        data.to_csv(data,
                    sep=",",
                    decimal=".",
                    )
        pass

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
        return None