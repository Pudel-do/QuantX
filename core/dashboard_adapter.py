from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output, State, ALL
import plotly.express as px
import plotly.graph_objects as go

import pandas as pd
import numpy as np
import threading
import webbrowser
import logging

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

from core.portfolio_generator import PortfolioGenerator


# ======================================================================================
# DASHBOARD ADAPTER
# ======================================================================================

class DashboardAdapter:
    def __init__(
        self,
        assets,
        ticks,
        tick_mapping,
        moving_avg,
        opt_moving_avg,
        port_types,
        stock_rets,
        bench_rets,
        stock_infos,
        fundamentals,
        model_backtest,
        model_validation,
        models,
        model_data,
        actual_quotes,
    ):
        # ---------------- Dash App ----------------
        self.app = Dash(__name__)
        self.app.title = "Financial Dashboard"

        # ---------------- Static Inputs ----------------
        self.assets = assets
        self.ticks = ticks
        self.tick_mapping = tick_mapping
        self.models = models
        self.model_data = model_data
        self.port_types = port_types

        # ---------------- Config ----------------
        self.params = read_json("parameter.json")
        const = read_json("constant.json")
        self.const_cols = const["columns"]
        self.fundamental_cols = const["fundamentals"]["measures"]

        self.weight_list = [
            self.const_cols["opt_weight"],
            self.const_cols["act_weight"],
        ]

        # ---------------- Data Preparation ----------------
        self.moving_avg = rename_dataframe(moving_avg, tick_mapping)
        self.opt_moving_avg = rename_dataframe(opt_moving_avg, tick_mapping)
        self.stock_rets = rename_dataframe(stock_rets, tick_mapping)
        self.bench_rets = rename_dataframe(bench_rets, tick_mapping)
        self.stock_infos = rename_dataframe(stock_infos, tick_mapping)
        self.fundamentals = rename_dataframe(fundamentals, tick_mapping)

        self.model_backtest = rename_dictionary(model_backtest, tick_mapping)
        self.model_validation = rename_dictionary(model_validation, tick_mapping)
        self.actual_quotes = rename_dictionary(actual_quotes, tick_mapping)

        # ---------------- Cached Time Ranges ----------------
        self.quote_marks, self.quote_range = self._init_time_range_values(self.moving_avg)
        self.rets_marks, self.rets_range = self._init_time_range_values(self.stock_rets)

        # ---------------- Layout & Callbacks ----------------
        self._build_layout()
        self._register_callbacks()

    # ==================================================================================
    # LAYOUT
    # ==================================================================================

    def _build_layout(self):
        self.app.layout = html.Div(
            style={"maxWidth": "1600px", "margin": "auto"},
            children=[
                html.H1("📊 Financial Dashboard", style={"textAlign": "center"}),

                self._analysis_section(),
                html.Hr(),

                self._model_section(),
                html.Hr(),

                self._portfolio_section(),
            ],
        )

    def _analysis_section(self):
        return html.Div(
            [
                html.H2("Technical Analysis"),

                dcc.Dropdown(
                    id="tick_dropdown_analysis",
                    options=[{"label": a, "value": a} for a in self.assets],
                    value=self.assets[0],
                    clearable=False,
                ),

                dcc.RangeSlider(
                    id="time_range_slider_quote",
                    min=0,
                    max=len(self.quote_range) - 1,
                    value=[0, len(self.quote_range) - 1],
                    marks=self.quote_marks,
                    allowCross=False,
                ),

                dcc.Graph(id="quote_ma_line"),
                dcc.Graph(id="ma_performance_line"),
                dcc.Graph(id="return_hist"),

                html.H3("Fundamentals"),
                dcc.Checklist(
                    id="checklist_fundamentals",
                    options=[{"label": c, "value": c} for c in self.fundamental_cols],
                    value=[self.fundamental_cols[0]],
                    inline=True,
                ),
                dcc.Graph(id="fundamentals_bar"),

                html.H3("Stock Infos"),
                dcc.Checklist(
                    id="checklist_stock_infos",
                    options=[{"label": c, "value": c} for c in self.stock_infos.columns],
                    value=[self.params["stock_infos"][0]],
                    inline=True,
                ),
                dcc.Graph(id="stock_infos_bar"),

                html.H3("Market Overview"),
                dcc.RangeSlider(
                    id="time_range_slider_returns",
                    min=0,
                    max=len(self.rets_range) - 1,
                    value=[0, len(self.rets_range) - 1],
                    marks=self.rets_marks,
                    allowCross=False,
                ),
                dcc.Graph(id="cumulated_stock_returns"),
                dash_table.DataTable(id="stock_performance_table"),
                dcc.Graph(id="corr_heatmap"),
            ]
        )

    def _model_section(self):
        return html.Div(
            [
                html.H2("Backtesting of Forecast Models"),

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
            ]
        )

    def _portfolio_section(self):
        return html.Div(
            [
                html.H2("Portfolio Analysis"),

                dcc.Dropdown(
                    id="weight_filter",
                    options=[{"label": w, "value": w} for w in self.weight_list],
                    value=self.const_cols["opt_weight"],
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

                dcc.Checklist(
                    id="portfolio_checklist",
                    options=[{"label": v, "value": v} for v in self.port_types.values()],
                    value=[list(self.port_types.values())[0]],
                    inline=True,
                ),

                dcc.RangeSlider(
                    id="time_range_slider_port",
                    min=0,
                    max=len(self.rets_range) - 1,
                    value=[0, len(self.rets_range) - 1],
                    marks=self.rets_marks,
                    allowCross=False,
                ),

                dcc.Graph(id="portfolio_performances"),
                dash_table.DataTable(id="performance_table"),
                dash_table.DataTable(id="weight_table"),

                html.H3("Long Positions"),
                dcc.Dropdown(
                    id="portfolio_dropdown",
                    options=[{"label": v, "value": v} for v in self.port_types.values()],
                    value=list(self.port_types.values())[0],
                ),
                dash_table.DataTable(id="long_positions"),
            ]
        )

    # ==================================================================================
    # CALLBACKS
    # ==================================================================================

    def _register_callbacks(self):

        # ---------- Technical Analysis ----------
        @self.app.callback(
            Output("quote_ma_line", "figure"),
            Output("ma_performance_line", "figure"),
            Output("return_hist", "figure"),
            Input("tick_dropdown_analysis", "value"),
            Input("time_range_slider_quote", "value"),
        )
        def update_analysis_charts(tick, slider):
            df = self._filter_df(self.moving_avg, tick)
            df, start, end = self._filter_time_range(df, slider)

            quote = df[self.const_cols["quote"]].ffill()
            sma1 = df[self.const_cols["sma1"]]
            sma2 = df[self.const_cols["sma2"]]

            rets = calculate_returns(quote)

            quote_fig = px.line(
                x=df.index,
                y=[quote, sma1, sma2],
                title=f"{tick} – Price & Moving Averages",
                template="plotly_white",
            )

            perf_fig = px.line(
                df,
                y=[self.const_cols["cumreturns"], self.const_cols["cumstrategy"]],
                title="Strategy vs Market",
                template="plotly_white",
            )

            hist_fig = px.histogram(
                rets,
                nbins=40,
                title="Return Distribution",
                template="plotly_white",
            )

            return quote_fig, perf_fig, hist_fig

        # ---------- Fundamentals ----------
        @self.app.callback(
            Output("fundamentals_bar", "figure"),
            Input("tick_dropdown_analysis", "value"),
            Input("checklist_fundamentals", "value"),
        )
        def update_fundamentals(tick, cols):
            df = self._filter_df(self.fundamentals, tick)
            if df.empty:
                df = pd.DataFrame(columns=cols)
            return px.bar(df[cols], template="plotly_white")

        @self.app.callback(
            Output("stock_infos_bar", "figure"),
            Input("checklist_stock_infos", "value"),
        )
        def update_stock_infos(cols):
            return px.bar(self.stock_infos[cols], template="plotly_white")

        # ---------- Market Overview ----------
        @self.app.callback(
            Output("cumulated_stock_returns", "figure"),
            Output("stock_performance_table", "data"),
            Output("corr_heatmap", "figure"),
            Input("time_range_slider_returns", "value"),
        )
        def update_market_overview(slider):
            df, start, end = self._filter_time_range(self.stock_rets, slider)

            cum = cumulate_returns(df).ffill()
            fig = px.line(
                cum,
                title=f"Cumulative Returns {start.date()} → {end.date()}",
                template="plotly_white",
            )

            corr = px.imshow(df.corr(), text_auto=True, template="plotly_white")

            perf = pd.DataFrame(index=self.assets)
            for c in df:
                perf.loc[c, self.const_cols["total_ret"]] = calc_total_return(df[c]) * 100
                perf.loc[c, self.const_cols["ann_mean_ret"]] = calc_annualized_mean_return(df[c]) * 100
                perf.loc[c, self.const_cols["ann_vola"]] = calc_annualized_vola(df[c])

            return fig, perf.round(2).reset_index().to_dict("records"), corr

        # ---------- Backtesting ----------
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
                    fig.add_trace(go.Scatter(x=df.index, y=df[m], name=m))
            fig.update_layout(template="plotly_white")
            return fig

        @self.app.callback(
            Output("validation_table", "data"),
            Input("tick_dropdown_models", "value"),
        )
        def update_validation(tick):
            df = self.model_validation.get(tick, pd.DataFrame())
            return df.round(3).reset_index().to_dict("records")

    # ==================================================================================
    # UTILS
    # ==================================================================================

    def run(self, debug=True):
        def run_dash():
            self.app.run_server(debug=debug, use_reloader=False)

        threading.Thread(target=run_dash).start()
        webbrowser.open_new("http://127.0.0.1:8050/")

    def _init_time_range_values(self, ts):
        dates = ts.index.unique()
        marks = {}
        seen = set()
        for i, d in enumerate(dates):
            y = str(d.year)
            if y not in seen:
                marks[i] = y
                seen.add(y)
        return marks, dates

    def _filter_df(self, df, tick):
        try:
            mask = df[self.const_cols["ticker"]] == tick
            return df[mask].drop(columns=self.const_cols["ticker"])
        except Exception:
            return pd.DataFrame()

    def _filter_time_range(self, data, slider):
        start = data.index[slider[0]]
        end = data.index[slider[1]]
        return data.loc[start:end], start, end
