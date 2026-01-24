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
from dashboard.visuals.annotations import (
    build_dash_annotation,
    build_plotly_annotation
)

from core.portfolio_generator import PortfolioGenerator

class DashboardAdapter:

    def __init__(self):
        self.const_cols = read_json("constant.json")["dashboard"]

    def build_cum_returns(self, slider):
        rets_pivot = self._load_pivot_returns(slider)

        cum_rets = rets_pivot.cumsum().apply(np.exp)
        cum_rets = cum_rets * 1000
        fig = px.line(
            cum_rets,
            x=cum_rets.index,
            y=cum_rets.columns
        )

        start, end = self.slider_to_dates(slider)
        fig.add_annotation(
            **build_plotly_annotation(start, end)
        )

        fig.update_layout(
            title="Kumulierte Renditen",
            hovermode="x unified",
            title_x=0.5,
        )

        fig.update_xaxes(title_text=f"{self.const_cols["date"]}", tickformat="%Y-%m-%d")
        fig.update_yaxes(title_text=f"{self.const_cols["cum_ret"]}", tickformat=".2f")

        return fig
    
    def build_return_peformance(self, slider):
        rets_pivot = self._load_pivot_returns(slider)

        rows = []
        for asset in rets_pivot.columns:
            values = rets_pivot[asset].dropna()
            rows.append({
                self.const_cols["asset"]: asset,
                self.const_cols["ann_mean_ret"]: calc_annualized_mean_return(values) * 100,
                self.const_cols["total_ret"]: calc_total_return(values) * 100,
                self.const_cols["ann_vola"]: calc_annualized_vola(values)
            })
        performance_table = pd.DataFrame(rows).round(2).to_dict("records")
        return performance_table
    
    @lru_cache(maxsize=32)
    def build_corr_heatmap(self, slider):
        pass

    @lru_cache(maxsize=32)
    def _load_pivot_returns(self, slider):
        start, end = self.slider_to_dates(slider)
 
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
        ).sort_index()
        
        return rets_pivot
    
    def slider_to_dates(self, slider):
        start_ts, end_ts = slider
        start = datetime.fromtimestamp(start_ts).strftime("%Y-%m-%d")
        end = datetime.fromtimestamp(end_ts).strftime("%Y-%m-%d")

        return start, end


    
