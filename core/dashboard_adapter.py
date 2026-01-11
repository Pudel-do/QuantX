import plotly.express as px
import plotly.graph_objects as go

import pandas as pd
import numpy as np
import logging

from functools import lru_cache
from datetime import datetime

from misc.utils import (
    read_json,
    rename_dataframe,
    rename_dictionary,
    calculate_returns,
    calc_annualized_mean_return,
    calc_total_return,
    calc_annualized_vola,
    cumulate_returns,
    get_future_returns
)

from misc.schema_utils import read_sql

from core.portfolio_generator import PortfolioGenerator

class DashboardAdapter:

    def __init__(self):
        pass

    @lru_cache(maxsize=32)
    def build_cum_returns(self, slider):

        start_ts, end_ts = slider
        start = datetime.fromtimestamp(start_ts).strftime("%Y-%m-%d")
        end = datetime.fromtimestamp(end_ts).strftime("%Y-%m-%d")
 
        query = """
        SELECT a.name, r.date, r.return
        FROM return_features as r
        JOIN assets as a
        ON r.asset_id = a.asset_id
        WHERE a.active_flag = 1
        AND a.benchmark_flag = 0
        AND r.date >= :start
        AND r.date <= :end
        """

        rets = read_sql(
            query=query,
            params={
                "start": start,
                "end": end
            }
        )

        rets_pivot = rets.pivot(
            index="date",
            columns="name",
            values="return"
        )

        cum_rets = rets_pivot.cumsum().apply(np.exp)
        fig = px.line(
            cum_rets,
            x=cum_rets.index,
            y=cum_rets.columns
        )

        return fig
