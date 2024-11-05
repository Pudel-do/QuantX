import numpy as np
import pandas as pd
import keras
import tensorflow as tf
import kerastuner as kt
import io
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error
from keras import layers
from core.base_model import BaseModel
from misc.misc import *

class OneStepLSTM(BaseModel):
    def __init__(self):
        super().__init__(model_name="OneStepLSTM")
        self._create_model_id()

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
        self.scaled_data = scaled_data
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        return None

    def build_model(self):
        """Function builds sequential model with LSTM layer
        and with subsequential compiling. Compiled model
        is inherited to model class

        :return: None
        :rtype: None
        """
        seq_length = self.params["sequence_length"]
        self.x_train, self.y_train = self._create_sequences(
            data=self.train_set,
            seq_length=seq_length
        )
        self.x_val, self.y_val = self._create_sequences(
            data=self.val_set,
            seq_length=seq_length
        )
        self.x_test, self.y_test = self._create_sequences(
            data=self.test_set,
            seq_length=seq_length
        )
        model = keras.Sequential()
        n_features = self.x_train.shape[2]
        input_shape = (seq_length, n_features)
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
        self.seq_length = seq_length
        self.n_features = n_features
        return None
    
    def hyperparameter_tuning(self):
        tuner = kt.RandomSearch(
            self._build_model_hp,
            objective='val_loss',
            max_trials=10,
            executions_per_trial=2,
            overwrite=True            
        )
        tuner.search(
            self.x_train, 
            self.y_train, 
            epochs=20, 
            validation_data=(self.x_val, self.y_val), 
            batch_size=32
        )
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        best_model = tuner.hypermodel.build(best_hps)
        self.model = best_model
        self.seq_length = best_hps.get('seq_length')

    def train(self):
        """Function trains LSTM model and saves callbacks
        in model logs for evaluation in tensorboard. 
        Trained model is inherited to model class

        :return: None
        :rtype: None
        """
        log_dir = get_log_path(
            ticker=self.ticker,
            model_id=self.model_id,
            log_key="training_logs"
        )
        tensorboard_cb = keras.callbacks.TensorBoard(log_dir)
        earlystop_cb = keras.callbacks.EarlyStopping(
            patience=self.params["patience"],
            restore_best_weights=True
        )
        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=self.params["epochs"],
            batch_size=32,
            validation_data=(self.x_val, self.y_val),
            callbacks=[tensorboard_cb, earlystop_cb],
        )
        return None
    
    def evaluate(self):
        """Function evaluates trained model on test set
        and writes performance results to evaluation logs.
        Test set performance can be analyzed in tensorboard.

        :return: None
        :rtype: None
        """
        #ToDO: Check transformation and lenght of x_test on main branch. Length must be equal to 90
        #ToDO: Problem probably while data scaling
        y_pred = self.model.predict(x=self.x_test, verbose=0)
        y_pred = y_pred[:, 0].reshape(self.pred_days, 1)
        target = self.y_test[:, 0].reshape(self.pred_days, 1)
        y_pred = self.target_scaler.inverse_transform(y_pred)
        target = self.target_scaler.inverse_transform(target)
        rmse = root_mean_squared_error(y_true=target, y_pred=y_pred)
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
        """Function performs one step ahead out-of-sample prediction 
        for defined prediction period in parameter.json based on the
        last sequence. Prediction values are inverse transformed
        to original scale and start by the first business day 
        following the last day from the test set.

        :return: Out-of-sample prediction
        :rtype: Series
        """
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
    
    def _build_model_hp(self, hp):
        """Function builds sequential model with LSTM layer
        and with subsequential compiling. Compiled model
        is inherited to model class

        :return: None
        :rtype: None
        """
        seq_length = hp.Int(
            'seq_length', 
            min_value=30, 
            max_value=120, 
            step=10)
        self.x_train, self.y_train = self._create_sequences(
            data=self.train_set,
            seq_length=seq_length
        )
        self.x_val, self.y_val = self._create_sequences(
            data=self.val_set,
            seq_length=seq_length
        )
        self.x_test, self.y_test = self._create_sequences(
            data=self.test_set,
            seq_length=seq_length
        )
        n_features = self.x_train.shape[2]
        input_shape = (seq_length, n_features)
        model = keras.Sequential()
        # model.add(layers.Input(shape=input_shape))
        for i in range(hp.Int('num_layers', 1, 3)):  # Tune between 1 and 3 layers
            if i == 0:
                # First layer with input shape based on the tuned sequence length
                model.add(layers.LSTM(units=hp.Int('units_' + str(i), min_value=32, max_value=128, step=32), 
                            return_sequences=True if hp.Int('num_layers', 1, 3) > 1 else False, 
                            input_shape=(input_shape)))
            else:
                # Additional layers
                model.add(layers.LSTM(units=hp.Int('units_' + str(i), min_value=32, max_value=128, step=32), 
                            return_sequences=False if i == hp.Int('num_layers', 1, 3) - 1 else True))
        model.add(layers.Dense(units=n_features))
        model.compile(
            optimizer="adam", 
            loss='mean_squared_error',
            metrics=[
                "root_mean_squared_error",
                "mean_absolute_error"
            ]
        )
        self.n_features = n_features
        return model
    
    def _create_sequences(self, data, seq_length):
        """Function creates sequences for defined
        period in parameter.json for LSTM model building
        separated by endogenous and exogenous variables
        for given model data

        :param data: Model data to create sequences for
        :type data: Array
        :return: Sequences for endogenous and exogenous variables
        :rtype: Array
        """
        exog, endog = [], []
        for i in range(seq_length, len(data)):
            exog.append(data[i-seq_length:i, :])
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
        tensorboard_cb = keras.callbacks.TensorBoard(log_dir)
        earlystop_cb = keras.callbacks.EarlyStopping(
            patience=self.params["patience"],
            restore_best_weights=True
        )
        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=self.params["epochs"],
            batch_size=32,
            validation_data=(self.x_val, self.y_val),
            callbacks=[tensorboard_cb, earlystop_cb],
        )
        return None
    
    def evaluate(self):
        y_pred = self.model.predict(x=self.x_test, verbose=0)
        y_pred = y_pred.reshape(self.pred_days, 1)
        target = self.y_test.reshape(self.pred_days, 1)
        y_pred = self.target_scaler.inverse_transform(y_pred)
        target = self.target_scaler.inverse_transform(target)
        rmse = root_mean_squared_error(y_true=target, y_pred=y_pred)
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