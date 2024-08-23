
import pandas as pd
import numpy as np
from core.file_adapter import FileAdapter
from core.dashboard_adapter import AnalysisDashboard
from misc.misc import *

def get_moving_average(quotes, ma_days):
    ma_dict = {}
    days_short = ma_days[0]
    days_long = ma_days[1]
    for name, values in quotes.items():
        values.name = "Quote"
        ma_short = values.ewm(
            halflife=0.5,
            min_periods=days_short
        ).mean()
        ma_short.name = f"MA_{days_short}"
        ma_long = values.ewm(
            halflife=0.5,
            min_periods=days_long
        ).mean()
        ma_long.name = f"MA_{days_long}"
        ma_df = pd.concat(
            objs=[values, ma_short, ma_long],
            axis=1,
            join="inner",
            ignore_index=False
        )
        ma_dict[name] = ma_df
    return ma_dict

def concat_ma_dict(ma_dict):
    for key, value in ma_dict.items():
        value["Ticker"] = key

    ma_concat = pd.concat(
        objs=ma_dict.values(),
        axis=0,
        ignore_index=False
    )
    return ma_concat
        


if __name__ == "__main__":
    parameter = read_json("parameter.json")["analysis"]
    ma_days = parameter["ma_days"]
    closing_quotes = FileAdapter().load_closing_quotes()
    moving_average_dict = get_moving_average(
        quotes=closing_quotes, 
        ma_days=ma_days
        )
    concat_moving_averages = concat_ma_dict(moving_average_dict)
    app = AnalysisDashboard(quotes=concat_moving_averages)
    app.run()
    print("Finished")

