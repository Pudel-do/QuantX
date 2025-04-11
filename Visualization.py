import pandas as pd
import numpy as np
import warnings
from core.finance_adapter import FinanceAdapter
from core.dashboard_adapter import DashboardAdapter
from core.file_adapter import FileAdapter
from misc.misc import *
warnings.filterwarnings('ignore')

def transform_df(df, index_name):
    """Function transforms given dataframe
    by setting index name with input value
    and reindexes dataframe by setting index
    as new column

    :param df: Dataframe to transform
    :type df: Dataframe
    :param index_name: Index name for transformation
    :type index_name: String
    :return: Transformed dataframe
    :rtype: Dataframe
    """
    df.index.name = index_name
    df = df.reset_index()
    df = df.round(3)
    return df

def transform_dict(dict, index_name):
    """Function transforms dataframes in 
    given dictionoary by applying transform
    function

    :param dict: Dictionary containing dataframes
    :type dict: Dictionary
    :param index_name: Index name for transformation
    :type index_name: String
    :return: Adjusted dictionary with transformed dataframes
    :rtype: Dictionary
    """
    dict_adj = {}
    for key, value in dict.items():
        value = transform_df(
            df=value,
            index_name=index_name
        )
        dict_adj[key] = value
    return dict_adj

def get_tick_mapping(stock_ticks, bench_tick):
    """Function creates mapping dictionary
    with tick values as keys and the company
    name as values

    :param ticks: Ticker symbols
    :type ticks: List
    :return: Dictionary for ticker mapping and
    list containig company names for given ticker symbols
    :rtype: Dictionary, List
    """
    ticker_mapping = {}
    company_names = []
    for tick in stock_ticks:
        company_name = FinanceAdapter(tick).get_company_name()
        ticker_mapping[tick] = company_name
        company_names.append(company_name)
    benchmark_name = FinanceAdapter(bench_tick).get_company_name()
    ticker_mapping[bench_tick] = benchmark_name
    return ticker_mapping, company_names

def rename_dataframe(df, tick_map):
    df_cols = df.columns
    if CONST_COLS["ticker"] in df_cols:
        df_adj = df.replace(
            {CONST_COLS["ticker"]: tick_map}
        )
    else:
        df_adj = df.rename(mapper=tick_map, axis=0)
        df_adj = df_adj.rename(mapper=tick_map, axis=1)
    return df_adj

def rename_dictionary(dict, tick_map):
    dict_adj = {}
    for key, value in dict.items():
        if key in list(tick_map.keys()):
            key_adj = tick_map[key]
        else:
            key_adj = key
        value_adj = rename_dataframe(
            df=value,
            tick_map=tick_map
        )
        dict_adj[key_adj] = value_adj
    return dict_adj

if __name__ == "__main__":
    PARAMETER = read_json("parameter.json")
    CONST_COLS = read_json("constant.json")["columns"]
    CONST_DATA = read_json("constant.json")["datamodel"]
    ticks = PARAMETER["ticker"]
    bench_tick = PARAMETER["benchmark_tick"]
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
    bench_rets = FileAdapter().load_dataframe(
        path=CONST_DATA["raw_data_dir"],
        file_name=CONST_DATA["benchmark_returns_file"]
    )
    future_stock_rets = FileAdapter().load_dataframe(
        path=CONST_DATA["processed_data_dir"],
        file_name=CONST_DATA["future_stock_returns_file"]
    )
    stock_infos = FileAdapter().load_dataframe(
        path=CONST_DATA["raw_data_dir"],
        file_name=CONST_DATA["stock_infos"]
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
    stock_infos, _ = harmonize_tickers(object=stock_infos)
    stock_infos = stock_infos.transpose()
    ticker_mapping, companies = get_tick_mapping(
        stock_ticks=ticks,
        bench_tick=bench_tick
    )

    dashboard = DashboardAdapter(
        ids=ticks,
        ticks=ticks,
        moving_avg=moving_averages,
        opt_moving_avg=opt_moving_averages,
        stock_rets=stock_rets,
        bench_rets = bench_rets,
        stock_infos=stock_infos,
        fundamentals=fundamentals,
        model_backtest=model_backtest,
        model_validation=model_validation,
        models=models
    )
    dashboard.run(debug=True)


    