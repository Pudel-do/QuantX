import numpy as np
import pandas as pd
import keras
import tensorflow as tf
import io
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from keras import layers
from core.base_model import BaseModel
from misc.misc import *

class lstm_os(BaseModel):
    def __init__(self, ticker):
        super().__init__(model_name="OneStepLSTM", ticker=ticker)
        self._create_model_id()
        self.seq_length = self.params["sequence_length"]

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
        log_dir = self._get_log_path(log_key="training_logs")
        tb_callbacks = keras.callbacks.TensorBoard(log_dir)
        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=self.params["epochs"],
            batch_size=32,
            validation_data=(self.x_test, self.y_test),
            callbacks=[tb_callbacks],
        )
        return None
    
    def evaluate(self):
        y_pred = self.model.predict(x=self.x_test)
        y_pred = y_pred[:, 0].reshape(-1, 1)
        target = self.y_test[:, 0].reshape(-1, 1)
        y_pred = self.target_scaler.inverse_transform(y_pred)
        target = self.target_scaler.inverse_transform(target)
        mse = mean_squared_error(y_true=target, y_pred=y_pred)
        rmse = np.sqrt(mse)
        rmse = np.round(rmse, 3)
        log_dir = self._get_log_path(log_key="evaluation_logs")
        file_writer = tf.summary.create_file_writer(log_dir)
        figure = self._create_plot(
            target=target, 
            y_pred=y_pred, 
            rmse=rmse
        )
        with file_writer.as_default():
            tf.summary.image(
                "Real vs Predicted", 
                self._plot_to_image(figure), 
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
    
    def _create_plot(self, target, y_pred, rmse):
        fig, ax = plt.subplots()
        ax.plot(target, label="Real Values", color='blue')
        ax.plot(y_pred, label="Predicted Values", color='red')
        ax.legend()
        ax.set_title(f"In-Sample prediction for {self.ticker} with RMSE of {rmse}")
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Value")
        return fig
    
    def _plot_to_image(self, figure):
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(figure)
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        return image


    def _get_log_path(self, log_key):
        """Function builds path for saving model logs.
        Underlying directory elements are defined in 
        constant.json

        :return: Path for saving model logs
        :rtype: String
        """
        data_model = read_json("constant.json")["datamodel"]
        models_dir = data_model["models_dir"]
        log_dir = data_model[log_key]
        path = os.path.join(
            os.getcwd(), 
            models_dir,
            self.ticker,
            log_dir, 
            self.model_id
        )
        return path

