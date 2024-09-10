import numpy as np
from abc import ABC, abstractmethod
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from misc.misc import *

class BaseModel(ABC):
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None
        self.data = None
        self.params = read_json("parameter.json")

    @abstractmethod
    def preprocess_data(self):
        pass

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def train_model(self):
        pass

    def load_data(self, data):
        self.data = data

    def _train_test_split(self, endog, exog):
        train_ratio = self.params["train_ratio"]
        train_size = int(len(endog) * train_ratio)
        x_train, x_test = exog[:train_size], exog[train_size:]
        y_train, y_test = endog[:train_size], endog[train_size:]
        return x_train, x_test, y_train, y_test
    
    def _data_scaling(self, data):
        """Function scales model data for target and 
        feature variables with given scale method.
        The function differentiates between target
        column and non-target colums to inherit the 
        target scaler to model class for later inverse
        transformation

        :param data: Dataframe containig model data
        for target and feature variables
        :type data: Dataframe
        :return: Scaled model data
        :rtype: Array
        """
        target_scaler = MinMaxScaler(feature_range=(0,1))
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data_list = []
        for col, ser in data.items():
            arr = ser.values.reshape(-1, 1)
            if col == self.params["target_col"]:
                scaled_arr = target_scaler.fit_transform(arr)
                self.target_scaler = target_scaler
            else:
                scaled_arr = scaler.fit_transform(arr)
            scaled_data_list.append(scaled_arr)
        scaled_data_tuple = tuple(scaled_data_list)
        scaled_data = np.hstack(scaled_data_tuple)
        return scaled_data

