import numpy as np
from datetime import datetime
from abc import ABC, abstractmethod
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from misc.misc import *

class BaseModel(ABC):
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None
        self.data = None
        self.params = read_json("parameter.json")
        self.pred_days = self.params["prediction_days"]

    def init_data(self, data, ticker):
        """Function initializes model data and respective ticker

        :param data: Full time series of endogenous and
        exogenous variabels for model building
        :type data: Series
        :param ticker: Stock ticker
        :type ticker: String
        """
        self.data = data
        self.ticker = ticker

    def _data_split(self, data, seq_length, use_val_set):
        """Function splits data set into train, test
        and validation set. Test set consits of the last
        n observations with n equals the out-of-sample
        prediction period.

        :param data: Data set of endogenous and 
        exogenous variables for model building
        :type data: Array
        :param seq_length: Sequence length for LSTM models
        :type seq_length: Integer
        :param use_val_set: Flag whether to return validation set
        :type use_val_set: Boolean
        :return: Train, validation and test set
        :rtype: Array
        """
        test_size = self.params["prediction_days"] + seq_length
        test_set = data[-test_size:]
        train_val_data = data[:-test_size]
        if use_val_set:
            train_size = int(len(train_val_data) * self.params["train_ratio"])
            train_set = train_val_data[:train_size]
            val_set = train_val_data[train_size:]
            return train_set, val_set, test_set
        else:
            train_set = train_val_data
            return train_set, test_set
    
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
    
    def _create_model_id(self):
        """Function defines model id for later reference.
        Model id consists of actual timestamp and
        model class name and is inherited to model class

        :return: None
        :rtype: None
        """
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        model_name = self.model_name
        model_id = f"{timestamp}_{model_name}"
        self.model_id = model_id
        return None
    
    @abstractmethod
    def preprocess_data(self):
        pass

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def hyperparameter_tuning(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @abstractmethod
    def predict(self):
        pass


