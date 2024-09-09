from abc import ABC, abstractmethod
from misc.misc import *

class BaseModel(ABC):
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None
        self.data = None
        self.params = read_json("parameter.json")

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def preprocess_data(self):
        pass

    def load_data(self, data):
        self.data = data

    def train_test_split(self, endog, exog):
        train_ratio = self.params["train_size"]
        train_size = int(len(endog) * train_ratio)
        if exog is None:
            x_train, x_test = endog[:train_size], endog[train_size:]
            return x_train, x_test, x_train, x_test
        else:
            x_train, x_test = exog[:train_size], exog[train_size:]
            y_train, y_test = endog[:train_size], endog[train_size:]
            return x_train, x_test, y_train, y_test

    def train(self):
        pass