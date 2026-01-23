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
        cum_rets = cum_rets * 1000
        fig = px.line(
            cum_rets,
            x=cum_rets.index,
            y=cum_rets.columns
        )

        fig.add_annotation(
            text=f"<b>Zeitraum</b><br>{start} – {end}",
            xref="paper",
            yref="paper",
            x=0.01,
            y=0.99,
            xanchor="left",
            yanchor="top",
            showarrow=False,
            font=dict(
                size=14,
                color="black"
            ),
            bgcolor="rgba(255,255,255,0.6)",
            bordercolor="rgba(0,0,0,0.15)",
            borderwidth=0
        )

        fig.update_layout(
            title="Kumulierte Renditen",
            hovermode="x unified",
            title_x=0.5,
        )

        fig.update_xaxes(title_text="Datum", tickformat="%Y-%m-%d")
        fig.update_yaxes(title_text="Kumulierte Rendite", tickformat=".2f")

        return fig
    
    @lru_cache(maxsize=32)
    def build_corr_heatmap(self, slider):
        pass
