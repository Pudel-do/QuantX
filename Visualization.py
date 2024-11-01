import pandas as pd
import numpy as np
from core.dashboard_adapter import DashboardAdapter
from core.file_adapter import FileAdapter
from misc.misc import *

def preprocess_df(df, index_name):
    df.index.name = index_name
    df = df.reset_index()
    df = df.round(3)
    return df

def preprocess_dict(dict, index_name):
    dict_adj = {}
    for key, value in dict.items():
        value.index.name = index_name
        value = value.reset_index()
        value = value.round(3)
        dict_adj[key] = value
    return dict_adj


if __name__ == "__main__":
    PARAMETER = read_json("parameter.json")
    CONST_COLS = read_json("constant.json")["columns"]
    CONST_DATA = read_json("constant.json")["datamodel"]
    moving_averages = FileAdapter().load_dataframe(
        path=CONST_DATA["processed_data_dir"],
        file_name=CONST_DATA["moving_averages_file"]
    )
    opt_moving_averages = FileAdapter().load_dataframe(
        path=CONST_DATA["processed_data_dir"],
        file_name=CONST_DATA["optimal_moving_averages_file"]
    )
    stock_rets = FileAdapter().load_dataframe(
        path=CONST_DATA["raw_data_dir"],
        file_name=CONST_DATA["stock_returns_file"]
    )
    fundamentals = FileAdapter().load_dataframe(
        path=CONST_DATA["processed_data_dir"],
        file_name=CONST_DATA["fundamentals_file"]
    )
    model_backtest = FileAdapter().load_object(
        path=CONST_DATA["processed_data_dir"],
        file_name=CONST_DATA["backtest_model_file"]
    )
    model_validation = FileAdapter().load_object(
        path=CONST_DATA["processed_data_dir"],
        file_name=CONST_DATA["validation_model_file"]
    )
    models = FileAdapter().load_object(
        path=CONST_DATA["processed_data_dir"],
        file_name=CONST_DATA["model_list"]
    )
    cum_bench_rets = FileAdapter().load_dataframe(
        path=CONST_DATA["processed_data_dir"],
        file_name=CONST_DATA["cum_benchmark_returns_file"]
    )
    cum_hist_rets = FileAdapter().load_dataframe(
        path=CONST_DATA["processed_data_dir"],
        file_name=CONST_DATA["cum_historical_returns_file"]
    )
    cum_future_rets = FileAdapter().load_dataframe(
        path=CONST_DATA["processed_data_dir"],
        file_name=CONST_DATA["cum_future_returns_file"]
    )
    port_performance = FileAdapter().load_dataframe(
        path=CONST_DATA["processed_data_dir"],
        file_name=CONST_DATA["port_performance_file"]
    )
    long_pos = FileAdapter().load_object(
        path=CONST_DATA["processed_data_dir"],
        file_name=CONST_DATA["long_position_file"]
    )
    port_types = FileAdapter().load_object(
        path=CONST_DATA["processed_data_dir"],
        file_name=CONST_DATA["port_types"]
    )
    port_performance = preprocess_df(
        df=port_performance,
        index_name=CONST_COLS["port_types"]
    )
    model_validation = preprocess_dict(
        dict=model_validation,
        index_name=CONST_COLS["measures"]
    )
    long_pos = preprocess_dict(
        dict=long_pos,
        index_name=CONST_COLS["ticker"]
    )
    dashboard = DashboardAdapter(
        moving_avg=moving_averages,
        opt_moving_avg=opt_moving_averages,
        stock_rets=stock_rets,
        fundamentals=fundamentals,
        model_backtest=model_backtest,
        model_validation=model_validation,
        models=models,
        cum_bench_rets=cum_bench_rets,
        cum_hist_rets=cum_hist_rets,
        cum_future_rets=cum_future_rets,
        port_performance=port_performance,
        long_pos=long_pos,
        port_types=port_types
    )
    dashboard.run(debug=True)


    