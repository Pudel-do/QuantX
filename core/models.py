import numpy as np
import pandas as pd
import keras
import tensorflow as tf
import kerastuner as kt
import io
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error
from pmdarima import auto_arima, ARIMA
from keras import layers
from keras import optimizers
from core.base_model import BaseModel
from misc.utils import *
import warnings
warnings.filterwarnings('ignore') 

class OneStepLSTM(BaseModel):
    def __init__(self):
        super().__init__(model_name="OneStepLSTM")
        self._create_model_id()
        self.earlystop_cb = keras.callbacks.EarlyStopping(
            patience=self.params["patience"],
            restore_best_weights=True
        )

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
        self.scaled_data = scaled_data
        return None

    def build_model(self):
        """Function builds sequential model with LSTM layer
        and with subsequential compiling. Compiled model
        is inherited to model class

        :return: None
        :rtype: None
        """
        exog_features = list(self.data.columns)
        last_model_obs = self.data.index[-1]
        seq_length = self.params["sequence_length"]
        train_set, val_set, test_set = self._data_split(
            data=self.scaled_data,
            seq_length=seq_length,
            use_val_set=True
        )
        self.val_set = val_set
        self.x_train, self.y_train = self._create_sequences(
            data=train_set,
            seq_length=seq_length
        )
        self.x_val, self.y_val = self._create_sequences(
            data=val_set,
            seq_length=seq_length
        )
        self.x_test, self.y_test = self._create_sequences(
            data=test_set,
            seq_length=seq_length
        )
        model = keras.Sequential()
        n_features = self.x_train.shape[2]
        input_shape = (seq_length, n_features)
        model.add(layers.Input(shape=input_shape))
        for i in range(self.params["n_layers"]):
            try:
                units = self.params["neurons"][i]
            except IndexError as e:
                logging.warning("Number of layers greater than list entries for neurons")
                units = 50
            try:
                drop_rate = self.params["drop_rates"][i]
            except IndexError as e:
                logging.warning("Number of layers greater than list entries for drop rates")
                drop_rate = 0
            
            return_sequences = i < (self.params["n_layers"] - 1)
            model.add(layers.LSTM(units=units, return_sequences=return_sequences))
            model.add(layers.Dropout(drop_rate))

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
        self.exog_features = exog_features
        self.last_model_obs = last_model_obs
        return None
    
    def hyperparameter_tuning(self):
        tuner = kt.Hyperband(
            self._build_model_hp,
            objective='val_loss',
            max_epochs=self.params["epochs"],
            factor=3,
            overwrite=True            
        )
        tuner.search(
            self.x_train, 
            self.y_train, 
            epochs=self.params["epochs"], 
            validation_data=(self.x_val, self.y_val), 
            batch_size=32,
            callbacks=[self.earlystop_cb]
        )
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        best_model = tuner.hypermodel.build(best_hps)
        self.best_hps = best_hps
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
        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=self.params["epochs"],
            batch_size=32,
            validation_data=(self.x_val, self.y_val),
            callbacks=[tensorboard_cb, self.earlystop_cb],
        )
        features = self.params["feature_cols"]
        lr = self.model.optimizer.learning_rate.value.name
        length_seq = self.model.input_shape[1]
        hidden_layers = 0
        neuron_list = []
        dropout_list = []
        for layer in self.model.layers:
            if isinstance(layer, layers.LSTM):
                hidden_layers += 1
                neuron_list.append(layer.units)
            elif isinstance(layer, layers.Dropout):
                dropout_list.append(layer.rate)
            else:
                pass
        model_facts = """
        Model Characteristics:
        - Exogenous features: {}
        - Sequence length: {}
        - Number of hidden layers: {}
        - Neurons: {}
        - Dropout rates: {}
        - Learning rate: {}
        """.format(
            list(features),
            length_seq,
            hidden_layers,
            neuron_list,
            dropout_list,
            lr
        )
        self.model_facts = model_facts
        return None
    
    def evaluate(self):
        """Function evaluates trained model on test set
        and writes performance results to evaluation logs.
        Test set performance can be analyzed in tensorboard.

        :return: None
        :rtype: None
        """
        y_pred_list = self._predict(
            base_data=self.val_set,
            forecast_horizon=len(self.y_test)
        )
        y_pred = np.array(y_pred_list).reshape(-1, 1)
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
            rmse=rmse,
            facts=self.model_facts
        )
        with file_writer.as_default():
            tf.summary.image(
                "Real vs Predicted", 
                plot_to_image(figure), 
                step=0
            )
        self.rmse = rmse
        return None
    
    def predict(self, actual_data=None):
        """Function performs one step ahead out-of-sample prediction 
        for defined prediction period in parameter.json based on the
        last sequence. Prediction values are inverse transformed
        to original scale and start by the first business day 
        following the last day from the test set.

        :return: Out-of-sample prediction
        :rtype: Series
        """
        if self._new_data_check(new_data=actual_data):
            pass

        self._refit_model(actual_data=actual_data)
        prediction_list = self._predict(
            base_data=self.scaled_data,
            forecast_horizon=self.pred_days
        )
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
    
    def _predict(self, base_data, forecast_horizon):
        """Function performs one step ahead out-of-sample
        forecast for given forecast horizon

        :param base_data: Data to generate sequence lenght from
        :type base_data: Dataframe
        :param forecast_horizon: Number of days to forecast
        :type forecast_horizon: Integer
        :return: Forecast values for target variable
        :rtype: List
        """
        last_seq = base_data[-self.seq_length:]
        last_seq = last_seq.reshape(1, self.seq_length, self.n_features)
        prediction_list = []
        for _ in range(forecast_horizon):
            prediction = self.model.predict(last_seq, verbose=0)
            prediction_list.append(prediction[0, 0])
            prediction = prediction.reshape(1, 1, self.n_features)
            cur_seq = last_seq[:, 1:, :]
            new_seq = np.append(cur_seq, prediction, axis=1)
            last_seq = new_seq
            
        return prediction_list
    
    def _build_model_hp(self, hp):
        """Function builds sequential model with LSTM layer
        and with subsequential compiling. Compiled model
        is inherited to model class

        :return: Model object from model building
        :rtype: LSTM object
        """
        seq_length = hp.Int(
            'seq_length', 
            min_value=10, 
            max_value=90, 
            step=10)
        train_set, val_set, test_set = self._data_split(
            data=self.scaled_data,
            seq_length=seq_length,
            use_val_set=True
        )
        self.val_set = val_set
        self.x_train, self.y_train = self._create_sequences(
            data=train_set,
            seq_length=seq_length
        )
        self.x_val, self.y_val = self._create_sequences(
            data=val_set,
            seq_length=seq_length
        )
        self.x_test, self.y_test = self._create_sequences(
            data=test_set,
            seq_length=seq_length
        )
        n_features = self.x_train.shape[2]
        input_shape = (seq_length, n_features)
        num_layers = hp.Int('num_layers', min_value=1, max_value=4, step=1)
        model = keras.Sequential()
        model.add(layers.Input(shape=input_shape))
        for i in range(num_layers):
            units = hp.Int(
                f'units_{i+1}', 
                min_value=32, 
                max_value=256, 
                step=32
            )
            dropout_rate = hp.Float(
                f'dropout_rate_{i+1}', 
                min_value=0.0, 
                max_value=0.5, 
                step=0.1
            )
            return_sequences = i < (num_layers - 1)
            model.add(layers.LSTM(
                units=units, 
                return_sequences=return_sequences,
                dropout=dropout_rate
                )
            )
        model.add(layers.Dense(units=n_features))
        optimizer = optimizers.Adam(
            learning_rate=hp.Float(
                'lr', 
                min_value=1e-4, 
                max_value=1e-2, 
                sampling='LOG'
            )
        )
        model.compile(
            optimizer=optimizer, 
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
    
    def _new_data_check(self, new_data):
        if new_data is None:
            return False
        try:
            return new_data.index[-1] > self.data.index[-1]
        except Exception as e:
            logging.error(f"Error while comparing refit data: {e}")
            return False
    
    def _refit_model(self, actual_data):
        logging.info("Refitting model with new data")
        #ToDO: Build error handling when self.data columns is less than self.exog_features
        self.data = actual_data[self.exog_features]
        self.preprocess_data()
        train_set, val_set, _ = self._data_split(
            data=self.scaled_data,
            seq_length=self.seq_length,
        use_val_set=True
        )
        x_train, y_train = self._create_sequences(train_set, self.seq_length)
        x_val, y_val = self._create_sequences(val_set, self.seq_length)

        self.model.fit(
            x_train,
            y_train,
            epochs=self.params["refit_epochs"],
            batch_size=32,
            validation_data=(x_val, y_val),
            callbacks=[self.earlystop_cb]
        )
        logging.info("Refitting finished")
        return None

class MultiStepLSTM(BaseModel):
    def __init__(self):
        super().__init__(model_name="OneStepLSTM")
        self._create_model_id()
        self.earlystop_cb = keras.callbacks.EarlyStopping(
            patience=self.params["patience"],
            restore_best_weights=True
        )

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
        self.scaled_data = scaled_data
        return None

    def build_model(self):
        """Function builds sequential model with LSTM layer
        and with subsequential compiling. Compiled model
        is inherited to model class

        :return: None
        :rtype: None
        """
        seq_length = self.params["sequence_length"]
        train_set, val_set, test_set = self._data_split(
            data=self.scaled_data,
            seq_length=seq_length,
            use_val_set=True
        )
        self.val_set = val_set
        self.x_train, self.y_train = self._create_sequences(
            data=train_set,
            seq_length=seq_length
        )
        self.x_val, self.y_val = self._create_sequences(
            data=val_set,
            seq_length=seq_length
        )
        self.x_test, self.y_test = self._create_sequences(
            data=test_set,
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
        self.seq_length = seq_length
        self.n_features = n_features
        return None
    
    def hyperparameter_tuning(self):
        tuner = kt.Hyperband(
            self._build_model_hp,
            objective='val_loss',
            max_epochs=self.params["epochs"],
            factor=3,
            overwrite=True            
        )
        tuner.search(
            self.x_train, 
            self.y_train, 
            epochs=self.params["epochs"], 
            validation_data=(self.x_val, self.y_val), 
            batch_size=32,
            callbacks=[self.earlystop_cb]
        )
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        best_model = tuner.hypermodel.build(best_hps)
        self.seq_length = best_hps.get('seq_length')
        self.model = best_model
        
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
        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=self.params["epochs"],
            batch_size=32,
            validation_data=(self.x_val, self.y_val),
            callbacks=[tensorboard_cb, self.earlystop_cb],
        )
        features = self.params["feature_cols"]
        lr = self.model.optimizer.learning_rate.value.name
        length_seq = self.model.input_shape[1]
        hidden_layers = 0
        neuron_list = []
        dropout_list = []
        for layer in self.model.layers:
            if isinstance(layer, layers.LSTM):
                hidden_layers += 1
                neuron_list.append(layer.units)
            elif isinstance(layer, layers.Dropout):
                dropout_list.append(layer.rate)
            else:
                pass
        model_facts = """
        Model Characteristics:
        - Exogenous features: {}
        - Sequence length: {}
        - Number of hidden layers: {}
        - Neurons: {}
        - Dropout rates: {}
        - Learning rate: {}
        """.format(
            list(features),
            length_seq,
            hidden_layers,
            neuron_list,
            dropout_list,
            lr
        )
        self.model_facts = model_facts
        return None
    
    def evaluate(self):
        """Function evaluates trained model on test set
        and writes performance results to evaluation logs.
        Test set performance can be analyzed in tensorboard.

        :return: None
        :rtype: None
        """
        y_pred = self._predict(base_data=self.val_set)
        y_pred = np.array(y_pred).reshape(-1, 1)
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
            rmse=rmse,
            facts=self.model_facts
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
        prediction = self._predict(base_data=self.scaled_data)
        prediction = np.array(prediction).reshape(-1, 1)
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
    
    def _predict(self, base_data):
        """Function performs one step ahead out-of-sample
        forecast for given forecast horizon

        :param base_data: Data to generate sequence lenght from
        :type base_data: Dataframe
        :param forecast_horizon: Number of days to forecast
        :type forecast_horizon: Integer
        :return: Forecast values for target variable
        :rtype: List
        """
        last_seq = base_data[-self.seq_length:]
        last_seq = last_seq.reshape(1, self.seq_length, self.n_features)
        prediction = self.model.predict(x=last_seq, verbose=0)
            
        return prediction
    
    def _build_model_hp(self, hp):
        """Function builds sequential model with LSTM layer
        and with subsequential compiling. Compiled model
        is inherited to model class

        :return: Model object from model building
        :rtype: LSTM object
        """
        seq_length = hp.Int(
            'seq_length', 
            min_value=10, 
            max_value=90, 
            step=10)
        train_set, val_set, test_set = self._data_split(
            data=self.scaled_data,
            seq_length=seq_length,
            use_val_set=True
        )
        self.x_train, self.y_train = self._create_sequences(
            data=train_set,
            seq_length=seq_length
        )
        self.x_val, self.y_val = self._create_sequences(
            data=val_set,
            seq_length=seq_length
        )
        self.x_test, self.y_test = self._create_sequences(
            data=test_set,
            seq_length=seq_length
        )
        n_features = self.x_train.shape[2]
        input_shape = (seq_length, n_features)
        num_layers = hp.Int('num_layers', min_value=1, max_value=4, step=1)
        model = keras.Sequential()
        model.add(layers.Input(shape=input_shape))
        for i in range(num_layers):
            units = hp.Int(
                f'units_{i+1}', 
                min_value=32, 
                max_value=256, 
                step=32
            )
            dropout_rate = hp.Float(
                f'dropout_rate_{i+1}', 
                min_value=0.0, 
                max_value=0.5, 
                step=0.1
            )
            return_sequences = i < (num_layers - 1)
            model.add(layers.LSTM(
                units=units, 
                return_sequences=return_sequences,
                dropout=dropout_rate
                )
            )
        model.add(layers.Dense(units=n_features))
        optimizer = optimizers.Adam(
            learning_rate=hp.Float(
                'lr', 
                min_value=1e-4, 
                max_value=1e-2, 
                sampling='LOG'
            )
        )
        model.compile(
            optimizer=optimizer, 
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
        for i in range(len(data)-seq_length-self.pred_days + 1):
            exog.append(data[i:i+seq_length, :])
            endog.append(data[i+seq_length:i+seq_length+self.pred_days, 0])

        return np.array(exog), np.array(endog)

class ArimaModel(BaseModel):
    def __init__(self):
        super().__init__(model_name="ArimaModel")
        self._create_model_id()

    def preprocess_data(self):
        """Function splits dataset into training and test set
        and separates both sets into endogenous and exogenous
        data arrays for model building
        :return: None
        :rtype: None
        """
        train_set, test_set = self._data_split(
            data=self.data,
            seq_length=0,
            use_val_set=False
        )
        x_train = train_set.iloc[:, 1:]
        x_train = self._empty_check(x_train)
        y_train = train_set.iloc[:, 0]
        x_test = test_set.iloc[:, 1:]
        x_test = self._empty_check(x_test)
        y_test = test_set.iloc[:, 0]
        y_full = pd.concat([y_train, y_test])
        try:
            x_full = pd.concat([x_train, x_test])
        except ValueError as e:
            x_full = None
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.x_full = x_full
        self.y_full = y_full
        return None

    def build_model(self):
        """Function builds seasonal ARIMA model on
        training set for predefined order. Compiled
        model is inheritated to model class

        :return: None
        :rtype: None
        """
        
        model = ARIMA(
            order=(
                self.params["order_p"],
                self.params["order_d"],
                self.params["order_q"]
            ),
            suppress_warnings=True,
        )
        self.model = model
        return None

    def hyperparameter_tuning(self):
        """Function uses auto arima algorithm to find the best 
        model order for optimizing the AIC criterion and builds
        model with best model order on the training set. Best model
        order and model are inheritated to the model class

        :return: None
        :rtype: None
        """
        model = auto_arima(
            y=self.y_train,
            X=self.x_train,
            start_p=0,
            start_q=0,
            d=None,
            m=1,
            test="kpss",
            max_order=None,
            stationary=False,
            seasonal=False,
            trace=True,
            suppress_warnings=True,
            stepwise=True
        )
        self.model = model
        return None
    
    def train(self):
        self.model.fit(
            y=self.y_train, 
            X=self.x_train
        )
        order = self.model.get_params()["order"]
        features = self.params["feature_cols"]
        model_facts = """
        Model Characteristics:
        - Exogenous features: {}
        - Model order: {}
        """.format(
            list(features),
            order
        )
        self.model_facts = model_facts
        return None

    def evaluate(self):
        test_length = len(self.y_test)
        y_pred = self.model.predict(
            n_periods=test_length,
            X=self.x_test,
        )
        y_pred = np.array(y_pred)
        target = np.array(self.y_test)
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
            rmse=rmse,
            facts=self.model_facts
        )
        with file_writer.as_default():
            tf.summary.image(
                "Real vs Predicted", 
                plot_to_image(figure), 
                step=0
            )
        return None

    def predict(self):
        self.model.fit(
            y=self.y_full,
            X=self.x_full
        )
        if self.x_full is None:
            future_exogs = None
        else:
            last_exogs = self.x_full.iloc[-1, :]
            last_exogs = pd.DataFrame(last_exogs)
            last_exogs = last_exogs.transpose()
            future_exogs = pd.concat(
                [last_exogs] * self.pred_days,
                ignore_index=True
            )
            future_exogs = np.array(future_exogs)
        prediction = self.model.predict(
            n_periods=self.pred_days,
            X=future_exogs,
        )
        prediction = np.array(prediction)
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
    
    def _empty_check(self, exog_data):
        """Functions checks given data to specify
        if data is empty and returns None if empty
        equals true

        :param exog_data: Exogenous model features
        :type exog_data: Dataframe
        :return:
        :rtype: 
        """
        if exog_data.empty:
            return None
        else:
            return exog_data
        
class DummyModel(BaseModel):
    def __init__(self):
        super().__init__(model_name="DummyModel")

    def predict(self):
        prediction = np.full((self.pred_days,), np.nan)
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
    
    def preprocess_data(self):
        pass

    def build_model(self):
        pass

    def hyperparameter_tuning(self):
        pass

    def train(self):
        pass

    def evaluate(self):
        pass