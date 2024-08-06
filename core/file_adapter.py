import pandas as pd
import os
from misc.misc import read_json

class DataConnector:
    def __init__(self) -> None:
        self.config = read_json("constant.json")["datamodel"]

    