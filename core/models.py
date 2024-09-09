import numpy as np
import pandas as pd
import keras
from sklearn.preprocessing import MinMaxScaler
from keras import layers
from core.base_model import BaseModel
from misc.misc import *

class lstm(BaseModel):
    def __init__(self):
        super().__init__(model_name="LSTM")

    def preprocess_data(self, seq_length):
        x, y = [], []
        for i in range(seq_length, len(self.data)):
            x.append(self.data.iloc[i-seq_length:i, :])
            y.append(self.data.iloc[i, 0])

        x = np.array(x)
        y = np.array(y)
        x_train, x_test, y_train, y_test = self.train_test_split(endog=y, exog=x)
        self.x_train = x_train
        self.x_test = x_test
        self.y_tain = y_train
        self.y_test = y_test
        pass

    def build_model(self):
        model = keras.Sequential()
        model.add(
            layers.LSTM(
                units=50, 
                return_sequences=True, 
                input_shape=(self.x_train.shape[1], 1)
            )
        )
        model.add(layers.LSTM(units=50))
        model.add(layers.Dense(units=1))
