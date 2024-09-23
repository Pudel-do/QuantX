import numpy as np
import pandas as pd
import keras
import tensorflow as tf
import io
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from keras import layers
from core.base_model import BaseModel
from misc.misc import *

class OneStepLSTM(BaseModel):
    def __init__(self):
        super().__init__(model_name="OneStepLSTM")
        self._create_model_id()
        self.seq_length = self.params["sequence_length"]

    def preprocess_data(self):
        """Function preprocesses raw model data for
        the underlying LSTM model. Preprocessing 
        includes data scaling, creating sequences for 
        LSTM modeling and data splitting into 
        train, validation and test sets for the target variable
        and feature columns. All relevant model buidling
        variables are inherited to the model class

        :param seq_length: Days to create sequences for
        :type seq_length: Integer
        :return: None
        :rtype: None
        """
        scaled_data = self._data_scaling(data=self.data)
        train_set, val_set, test_set = self._data_split(data=scaled_data)
        x_train, y_train = self._create_sequences(train_set)
        x_val, y_val = self._create_sequences(val_set)
        x_test, y_test = self._create_sequences(test_set)
        self.scaled_data = scaled_data
        self.x_train = x_train
        self.x_val = x_val
        self.x_test = x_test
        self.y_train = y_train
        self.y_val = y_val
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
        model.add(layers.Input(shape=input_shape))
        model.add(layers.LSTM(units=100, return_sequences=True))
        model.add(layers.Dropout(0.2))
        model.add(layers.LSTM(units=50, return_sequences=False))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(units=n_features))
        model.compile(
            optimizer="adam", 
            loss='mean_squared_error',
            metrics=[
                "root_mean_squared_error",
                "mean_absolute_error"
            ]
        )
        self.model = model
        self.n_features = n_features
        return None
    
    def train(self):
        """Function trains LSTM model and saves callbacks
        in model logs. Trained model is inherited to 
        model class

        :return: None
        :rtype: None
        """
        log_dir = get_log_path(
            ticker=self.ticker,
            model_id=self.model_id,
            log_key="training_logs"
        )
        tb_callbacks = keras.callbacks.TensorBoard(log_dir)
        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=self.params["epochs"],
            batch_size=32,
            validation_data=(self.x_val, self.y_val),
            callbacks=[tb_callbacks],
        )
        return None
    
    def evaluate(self):
        y_pred = self.model.predict(x=self.x_test, verbose=0)
        y_pred = y_pred[:, 0].reshape(self.pred_days, 1)
        target = self.y_test[:, 0].reshape(self.pred_days, 1)
        y_pred = self.target_scaler.inverse_transform(y_pred)
        target = self.target_scaler.inverse_transform(target)
        mse = mean_squared_error(y_true=target, y_pred=y_pred)
        rmse = np.sqrt(mse)
        rmse = np.round(rmse, 3)
        log_dir = get_log_path(
            ticker=self.ticker,
            model_id=self.model_id,
            log_key="evaluation_logs"
        )
        file_writer = tf.summary.create_file_writer(log_dir)
        figure = create_in_pred_fig(
            ticker=self.ticker,
            target=target,
            y_pred=y_pred,
            rmse=rmse
        )
        with file_writer.as_default():
            tf.summary.image(
                "Real vs Predicted", 
                plot_to_image(figure), 
                step=0
            )
        return None
    
    def predict(self):
        last_seq = self.scaled_data[-self.seq_length:]
        last_seq = last_seq.reshape(1, self.seq_length, self.n_features)
        prediction_list = []
        for _ in range(self.pred_days):
            prediction = self.model.predict(last_seq, verbose=0)
            prediction_list.append(prediction[0, 0])
            prediction = prediction.reshape(1, 1, self.n_features)
            cur_seq = last_seq[:, 1:, :]
            new_seq = np.append(cur_seq, prediction, axis=1)
            last_seq = new_seq
        
        prediction = np.array(prediction_list).reshape(-1, 1)
        prediction = self.target_scaler.inverse_transform(prediction)
        prediction = prediction.flatten()
        last_timestamp = self.data.index[-1]
        pred_start = last_timestamp + pd.DateOffset(days=1)
        pred_start = get_business_day(pred_start)
        prediction_index = pd.date_range(
            start=pred_start,
            periods=self.pred_days,
            freq="B",
            normalize=True
        )
        prediction = pd.Series(
            data=prediction,
            index=prediction_index
        )
        return prediction
    
    def _create_sequences(self, data):
        exog, endog = [], []
        for i in range(self.seq_length, len(data)):
            exog.append(data[i-self.seq_length:i, :])
            endog.append(data[i, :])

        return np.array(exog), np.array(endog)

class MultiStepLSTM(BaseModel):
    def __init__(self):
        super().__init__(model_name="MutliStepLSTM")
        self._create_model_id()
        self.seq_length = self.params["sequence_length"]

    def preprocess_data(self):
        """Function preprocesses raw model data for
        the underlying LSTM model. Preprocessing 
        includes data scaling, creating sequences for 
        LSTM modeling and data splitting into 
        train, validation and test sets for the target variable
        and feature columns. All relevant model buidling
        variables are inherited to the model class

        :param seq_length: Days to create sequences for
        :type seq_length: Integer
        :return: None
        :rtype: None
        """
        scaled_data = self._data_scaling(data=self.data)
        train_set, val_set, test_set = self._data_split(data=scaled_data)
        x_train, y_train = self._create_sequences(train_set)
        x_val, y_val = self._create_sequences(val_set)
        x_test, y_test = self._create_sequences(test_set)
        self.scaled_data = scaled_data
        self.x_train = x_train
        self.x_val = x_val
        self.x_test = x_test
        self.y_train = y_train
        self.y_val = y_val
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
        model.add(layers.Input(shape=input_shape))
        model.add(layers.LSTM(units=100, return_sequences=True))
        model.add(layers.Dropout(0.2))
        model.add(layers.LSTM(units=50, return_sequences=False))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(units=self.pred_days))
        model.compile(
            optimizer="adam", 
            loss='mean_squared_error',
            metrics=[
                "root_mean_squared_error",
                "mean_absolute_error"
            ]
        )
        self.model = model
        self.n_features = n_features
        return None
    
    def train(self):
        """Function trains LSTM model and saves callbacks
        in model logs. Trained model is inherited to 
        model class

        :return: None
        :rtype: None
        """
        log_dir = get_log_path(
            ticker=self.ticker,
            model_id=self.model_id,
            log_key="training_logs"
        )
        tb_callbacks = keras.callbacks.TensorBoard(log_dir)
        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=self.params["epochs"],
            batch_size=32,
            validation_data=(self.x_val, self.y_val),
            callbacks=[tb_callbacks],
        )
        return None
    
    def evaluate(self):
        y_pred = self.model.predict(x=self.x_test, verbose=0)
        y_pred = y_pred.reshape(self.pred_days, 1)
        target = self.y_test.reshape(self.pred_days, 1)
        y_pred = self.target_scaler.inverse_transform(y_pred)
        target = self.target_scaler.inverse_transform(target)
        mse = mean_squared_error(y_true=target, y_pred=y_pred)
        rmse = np.sqrt(mse)
        rmse = np.round(rmse, 3)
        log_dir = get_log_path(
            ticker=self.ticker,
            model_id=self.model_id,
            log_key="evaluation_logs"
        )
        file_writer = tf.summary.create_file_writer(log_dir)
        figure = create_in_pred_fig(
            ticker=self.ticker,
            target=target,
            y_pred=y_pred,
            rmse=rmse
        )
        with file_writer.as_default():
            tf.summary.image(
                "Real vs Predicted", 
                plot_to_image(figure), 
                step=0
            )
        return None
    
    def predict(self):
        last_seq = self.scaled_data[-self.seq_length:]
        last_seq = last_seq.reshape(1, self.seq_length, self.n_features)
        prediction = self.model.predict(x=last_seq, verbose=0)
        prediction = np.array(prediction).reshape(self.pred_days, 1)
        prediction = self.target_scaler.inverse_transform(prediction)
        prediction = prediction.flatten()
        last_timestamp = self.data.index[-1]
        pred_start = last_timestamp + pd.DateOffset(days=1)
        pred_start = get_business_day(pred_start)
        prediction_index = pd.date_range(
            start=pred_start,
            periods=self.pred_days,
            freq="B",
            normalize=True
        )
        prediction = pd.Series(
            data=prediction,
            index=prediction_index
        )
        return prediction
    
    def _create_sequences(self, data):
        exog, endog = [], []
        for i in range(len(data)-self.seq_length-self.pred_days + 1):
            exog.append(data[i:i+self.seq_length, :])
            endog.append(data[i+self.seq_length:i+self.seq_length+self.pred_days, 0])

        return np.array(exog), np.array(endog)