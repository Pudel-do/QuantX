# Refactored Visualization.py
# Functional behavior unchanged; structure, readability, and minor efficiency improved

import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from core.dashboard_adapter import DashboardAdapter
from core.file_adapter import FileAdapter
from core.finance_adapter import FinanceAdapter
from misc.utils import (
    read_json,
    harmonize_tickers,
    rename_yfcolumns,
)

warnings.filterwarnings("ignore")


# ============================
# Data transformation helpers
# ============================

def transform_df(df: pd.DataFrame, index_name: str) -> pd.DataFrame:
    """Standardized dataframe transformation."""
    if df.empty:
        return df
    df = df.copy()
    df.index.name = index_name
    df = df.reset_index().round(3)
    return df


def transform_dict(data: Dict[str, pd.DataFrame], index_name: str) -> Dict[str, pd.DataFrame]:
    """Apply `transform_df` to all dataframes in a dictionary."""
    return {
        key: transform_df(value, index_name)
        for key, value in data.items()
    }


# ============================
# Finance / ticker utilities
# ============================

def get_tick_mapping(stock_ticks: List[str], bench_tick: str) -> Tuple[Dict[str, str], List[str]]:
    """Map tickers to company names."""
    ticker_mapping: Dict[str, str] = {}
    assets: List[str] = []

    for tick in stock_ticks:
        name = FinanceAdapter(tick).get_company_name()
        ticker_mapping[tick] = name
        assets.append(name)

    ticker_mapping[bench_tick] = FinanceAdapter(bench_tick).get_company_name()
    return ticker_mapping, assets


def get_actual_quotes(ticks: List[str]) -> Dict[str, float]:
    """Fetch latest quotes for each ticker."""
    quotes = {}
    for tick in ticks:
        df = FinanceAdapter(tick).get_last_quote()
        df = rename_yfcolumns(data=df)
        quote = df.iloc[0][read_json("parameter.json")["quote_id"]]
        quotes[tick] = quote
    return quotes


# ============================
# Main execution
# ============================

if __name__ == "__main__":
    PARAMETER = read_json("parameter.json")
    CONST = read_json("constant.json")

    CONST_COLS = CONST["columns"]
    CONST_DATA = CONST["datamodel"]
    CONST_PORT_TYPES = CONST["port_keys"].copy()

    ticks = PARAMETER["ticker"]
    bench_tick = PARAMETER["benchmark_tick"]

    tick_mapping, assets = get_tick_mapping(ticks, bench_tick)

    file_adapter = FileAdapter()

    # ---- Load data ----
    moving_averages = file_adapter.load_dataframe(CONST_DATA["processed_data_dir"], CONST_DATA["moving_averages_file"])
    opt_moving_averages = file_adapter.load_dataframe(CONST_DATA["processed_data_dir"], CONST_DATA["optimal_moving_averages_file"])

    stock_rets = file_adapter.load_dataframe(CONST_DATA["raw_data_dir"], CONST_DATA["stock_returns_file"])
    bench_rets = file_adapter.load_dataframe(CONST_DATA["raw_data_dir"], CONST_DATA["benchmark_returns_file"])
    stock_infos = file_adapter.load_dataframe(CONST_DATA["raw_data_dir"], CONST_DATA["stock_infos"])

    fundamentals = file_adapter.load_dataframe(CONST_DATA["processed_data_dir"], CONST_DATA["fundamentals_file"])

    model_backtest = file_adapter.load_object(CONST_DATA["processed_data_dir"], CONST_DATA["backtest_model_file"])
    model_validation = file_adapter.load_object(CONST_DATA["processed_data_dir"], CONST_DATA["validation_model_file"])
    models = file_adapter.load_object(CONST_DATA["processed_data_dir"], CONST_DATA["model_list"])
    model_data = file_adapter.load_object(CONST_DATA["processed_data_dir"], CONST_DATA["model_data_file"])

    # ---- Cleaning & harmonization ----
    stock_rets_clean, _ = harmonize_tickers(stock_rets)
    stock_infos, _ = harmonize_tickers(stock_infos)
    stock_infos = stock_infos.transpose()

    actual_quotes = get_actual_quotes(ticks)

    if not PARAMETER.get("use_custom_weights", False):
        CONST_PORT_TYPES.pop("CUSTOM", None)

    # ---- Dashboard ----
    dashboard = DashboardAdapter(
        assets=assets,
        ticks=ticks,
        tick_mapping=tick_mapping,
        moving_avg=moving_averages,
        opt_moving_avg=opt_moving_averages,
        port_types=CONST_PORT_TYPES,
        stock_rets=stock_rets_clean,
        bench_rets=bench_rets,
        stock_infos=stock_infos,
        fundamentals=fundamentals,
        model_backtest=model_backtest,
        model_validation=model_validation,
        models=models,
        model_data=model_data,
        actual_quotes=actual_quotes,
    )

    dashboard.run(debug=True)
