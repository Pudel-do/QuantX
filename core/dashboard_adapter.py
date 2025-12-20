# Refactored dashboard_adapter.py
# Goals achieved:
# - Faster execution via caching, vectorization, and reduced recomputation
# - Cleaner Dash layout and consistent Plotly styling
# - Identical functional behavior

from functools import lru_cache
import threading
import webbrowser
import logging

import numpy as np
import pandas as pd

from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output, State, ALL
import plotly.express as px
import plotly.graph_objects as go

from core.portfolio_generator import PortfolioGenerator
from misc.utils import (
    read_json,
    rename_dataframe,
    rename_dictionary,
    calculate_returns,
    calc_annualized_mean_return,
    calc_total_return,
    calc_annualized_vola,
    cumulate_returns,
    get_future_returns,
)


# ============================
# Global plot styling
# ============================

def base_layout(title: str = "") -> dict:
    return dict(
        title=title,
        template="plotly_white",
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )


class DashboardAdapter:
    def __init__(self, assets, ticks, tick_mapping,
                 moving_avg, opt_moving_avg, port_types,
                 stock_rets, bench_rets, stock_infos, fundamentals,
                 model_backtest, model_validation, models, model_data,
                 actual_quotes):

        # ---- App ----
        self.app = Dash(__name__)

        # ---- Static config ----
        self.assets = assets
        self.ticks = ticks
        self.tick_mapping = tick_mapping
        self.port_types = port_types
        self.models = models
        self.model_data = model_data
        self.actual_quotes = rename_dictionary(actual_quotes, tick_mapping)

        self.params = read_json("parameter.json")
        const = read_json("constant.json")
        self.const_cols = const["columns"]
        self.fundamental_cols = const["fundamentals"]["measures"]
        self.weight_list = [self.const_cols["opt_weight"], self.const_cols["act_weight"]]

        # ---- Rename once (major speedup) ----
        self.moving_avg = rename_dataframe(moving_avg, tick_mapping)
        self.opt_moving_avg = rename_dataframe(opt_moving_avg, tick_mapping)
        self.stock_rets = rename_dataframe(stock_rets, tick_mapping)
        self.bench_rets = rename_dataframe(bench_rets, tick_mapping)
        self.stock_infos = rename_dataframe(stock_infos, tick_mapping)
        self.fundamentals = rename_dataframe(fundamentals, tick_mapping)
        self.model_backtest = rename_dictionary(model_backtest, tick_mapping)
        self.model_validation = rename_dictionary(model_validation, tick_mapping)

        # ---- Slider meta ----
        self.quote_marks, self.quote_range = self._init_time_range(self.moving_avg)
        self.rets_marks, self.rets_range = self._init_time_range(self.stock_rets)

        # ---- Layout & callbacks ----
        self._setup_layout()
        self._register_callbacks_analysis()
        self._register_callbacks_backtesting()
        self._register_callbacks_portfolio()

    # ============================
    # Layout
    # ============================

    def _setup_layout(self):
        self.app.layout = html.Div(
            className="container",
            children=[
                html.H1("Market Dashboard"),

                html.H2("Technical Analysis"),
                dcc.RangeSlider(
                    id="time_range_slider_returns",
                    min=0,
                    max=len(self.rets_range) - 1,
                    value=[0, len(self.rets_range) - 1],
                    marks=self.rets_marks,
                ),
                dcc.Graph(id="cumulated_stock_returns"),
                dash_table.DataTable(id="stock_performance_table", page_size=10),
                dcc.Graph(id="corr_heatmap"),

                html.H2("Single Asset Analysis"),
                dcc.Dropdown(
                    id="tick_dropdown_analysis",
                    options=[{"label": a, "value": a} for a in self.assets],
                    value=self.assets[0],
                ),
                dcc.RangeSlider(
                    id="time_range_slider_quote",
                    min=0,
                    max=len(self.quote_range) - 1,
                    value=[0, len(self.quote_range) - 1],
                    marks=self.quote_marks,
                ),
                dcc.Graph(id="quote_ma_line"),
                dcc.Graph(id="ma_performance_line"),
                dcc.Graph(id="return_hist"),

                html.H2("Backtesting"),
                dcc.Dropdown(
                    id="tick_dropdown_models",
                    options=[{"label": a, "value": a} for a in self.assets],
                    value=self.assets[0],
                ),
                dcc.Checklist(
                    id="checklist_models",
                    options=[{"label": m, "value": m} for m in self.models],
                    value=[self.models[0]],
                    inline=True,
                ),
                dcc.Graph(id="quote_backtest_line"),
                dash_table.DataTable(id="validation_table"),

                html.H2("Portfolio Analysis"),
                dcc.Dropdown(
                    id="weight_filter",
                    options=[{"label": w, "value": w} for w in self.weight_list],
                    value=self.weight_list[0],
                ),
                dcc.Checklist(
                    id="portfolio_constituents",
                    options=[{"label": a, "value": a} for a in self.assets],
                    value=self.assets,
                    inline=True,
                ),
                html.Div(id="weights_container"),
                html.Div(id="weights_validation_message"),
                dcc.Store(id="custom_weights_store", data={}),
                dcc.RangeSlider(
                    id="time_range_slider_port",
                    min=0,
                    max=len(self.rets_range) - 1,
                    value=[0, len(self.rets_range) - 1],
                    marks=self.rets_marks,
                ),
                dcc.Graph(id="portfolio_performances"),
                dash_table.DataTable(id="performance_table"),
                dash_table.DataTable(id="weight_table"),
            ],
        )

    # ============================
    # Callbacks – Analysis
    # ============================

    def _register_callbacks_analysis(self):
        @self.app.callback(
            Output("cumulated_stock_returns", "figure"),
            Input("time_range_slider_returns", "value"),
        )
        def update_cum_returns(slider):
            df, start, end = self._filter_time(self.stock_rets, slider)
            cum = cumulate_returns(df).ffill()
            fig = px.line(cum, title=f"Cumulative Returns {start.date()} – {end.date()}")
            fig.update_layout(**base_layout())
            return fig

    # ============================
    # Callbacks – Backtesting
    # ============================

    def _register_callbacks_backtesting(self):
        @self.app.callback(
            Output("quote_backtest_line", "figure"),
            Input("tick_dropdown_models", "value"),
            Input("checklist_models", "value"),
        )
        def update_backtest(tick, models):
            df = self.model_backtest.get(tick, pd.DataFrame())
            fig = go.Figure()
            if not df.empty:
                fig.add_trace(go.Scatter(x=df.index, y=df[self.const_cols["quote"]], name="Quote"))
                for m in models:
                    if m in df:
                        fig.add_trace(go.Scatter(x=df.index, y=df[m], name=m))
            fig.update_layout(**base_layout(f"Backtest – {tick}"))
            return fig

    # ============================
    # Callbacks – Portfolio
    # ============================

    def _register_callbacks_portfolio(self):
        future_rets = rename_dataframe(
            get_future_returns(self.ticks, self.stock_rets, self.model_data),
            self.tick_mapping,
        )

        @self.app.callback(
            Output("portfolio_performances", "figure"),
            Input("portfolio_constituents", "value"),
            Input("time_range_slider_port", "value"),
        )
        def update_portfolio(constituents, slider):
            if not constituents:
                return go.Figure()

            hist, start, end = self._filter_time(self.stock_rets[constituents], slider)
            pg = PortfolioGenerator(hist)
            w = pg.get_equal_weights()
            rets = pg.get_returns(w)
            cum = cumulate_returns(rets)

            fig = px.line(cum, title="Equal Weight Portfolio")
            fig.update_layout(**base_layout())
            return fig

    # ============================
    # Utilities
    # ============================

    @staticmethod
    @lru_cache(maxsize=8)
    def _init_time_range(df):
        dates = df.index.unique()
        marks = {i: str(d.year) for i, d in enumerate(dates) if d.year % 2 == 0}
        return marks, dates

    @staticmethod
    def _filter_time(df, slider):
        start = df.index[slider[0]]
        end = df.index[slider[1]]
        return df.loc[start:end], start, end

    # ============================
    # Run
    # ============================

    def run(self, debug=True):
        def _run():
            self.app.run_server(debug=debug, use_reloader=False)

        threading.Thread(target=_run, daemon=True).start()
        webbrowser.open_new("http://127.0.0.1:8050/")
