import numpy as np
import pandas as pd
import keras
import tensorflow as tf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras import layers
from core.base_model import BaseModel
from misc.misc import *

class lstm(BaseModel):
    def __init__(self):
        super().__init__(model_name="LSTM")

    def preprocess_data(self):
        """Function preprocesses raw model data for
        the underlying LSTM model. Preprocessing 
        includes data scaling, creating sequences for 
        LSTM modeling and data splitting into 
        train and test sets for the target variable
        and feature columns. All relevant model buidling
        variables are inherited to the model class

        :param seq_length: Days to create sequences for
        :type seq_length: Integer
        :return: None
        :rtype: None
        """
        scaled_data = self._data_scaling(data=self.data)
        seq_length = self.params["sequence_length"]
        exog, endog = [], []
        for i in range(seq_length, len(scaled_data)):
            exog.append(scaled_data[i-seq_length:i, :])
            endog.append(scaled_data[i, 0])

        exog = np.array(exog)
        endog = np.array(endog)
        x_train, x_test, y_train, y_test = self._train_test_split(endog=endog, exog=exog)
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        return None

    def build_model(self):
        """Function builds sequential model with LSTM layer
        and with subsequential compiling. Compiled model
        is inherited to model class

        :return: None
        :rtype: None
        """
        model = keras.Sequential()
        sequences = self.x_train.shape[1]
        n_features = self.x_train.shape[2]
        input_shape = (sequences, n_features)
        model.add(
            layers.LSTM(
                units=50, 
                return_sequences=True, 
                input_shape=input_shape
            )
        )
        model.add(layers.LSTM(units=50))
        model.add(layers.Dense(units=1))
        model.compile(
            optimizer="adam", 
            loss='mean_squared_error'
        )
        self.model = model
        return None
    
    def train_model(self):
        """Function trains LSTM model and saves callbacks
        in model logs. Trained model is inherited to 
        model class

        :return: None
        :rtype: None
        """
        run_logdir = self._get_log_path()
        tb_callbacks = keras.callbacks.TensorBoard(run_logdir)
        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=self.params["epochs"],
            validation_data=(self.x_test, self.y_test),
            callbacks=[tb_callbacks]
        )
        return None

    def _get_log_path(self):
        """Function builds path for saving model logs.
        Underlying directory elements are defined in 
        constant.json

        :return: Path for saving model logs
        :rtype: String
        """
        datamodel = read_json("constant.json")["datamodel"]
        models_dir = datamodel["models_dir"]
        log_dir = datamodel["model_logs"]
        file_name = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = os.path.join(os.getcwd(), models_dir, log_dir, file_name)
        return path

