from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
from core import logging_config
from misc.misc import read_json
import pandas as pd
import numpy as np
import time
import threading
import webbrowser
import logging
import os

class AnalysisDashboard:
    def __init__(self, ma_data, ma_values, returns, fundamentals):
        """
        :param ma_data: Dataframe with moving average values for each stock
        :type ma_data: Dataframe
        :param returns: Log returns for each stock
        :type returns: Dataframe
        :param fundamentals: Fundamental data for each stock
        :type fundamentals: Dataframe
        """
        self.ma_data = ma_data
        self.ma_values = ma_values
        self.returns = returns
        self.fundamentals = fundamentals
        self.tickers = read_json("parameter.json")["ticker"]
        self.const_cols = read_json("constant.json")["columns"]
        self.fundamental_list = read_json("constant.json")["fundamentals"]["measures"]
        self.app = Dash(__name__)
        self._setup_layout()
        self._register_callbacks()

    def _tick_filter(self, df, tick):
        """Function filters dataframe for given ticker symbol.
        If symbol is not in ticker column or ticker column
        does not exist, the functin returns an empty dataframe

        :param df: Fundamental data or quotes to filter 
        for given ticker symbol
        :type df: Dataframe
        :param tick: Ticker symbol for filtering
        :type tick: String
        :return: Filtered object
        :rtype: Dataframe
        """
        tick_col = self.const_cols["ticker"]
        try:
            filter_mask = df[tick_col] == tick
            df_filtered = df[filter_mask]
            df_filtered = df_filtered.drop(columns=tick_col)
            return df_filtered
        except KeyError:
            logging.error(f"Column {tick_col} or ticker symbol {tick} not in dataframe")
            return pd.DataFrame()

    def _setup_layout(self):
        """Function sets the layout for the dashboard app.
        All dashboard items like dropdowns, sliders and graphs 
        must be defined in this method
        """
        date_range = self.returns.index
        raw_marks = {i: str(date.year) \
            for i, date in enumerate(date_range)}
        seen_years = set()
        marks = {}
        for key, value in raw_marks.items():
            if value not in seen_years:
                marks[key] = value
                seen_years.add(value)
        self.app.layout = html.Div(
            [   
                html.H1("Technical and fundamental stock analysis"),
                dcc.Dropdown(
                    id="tick_dropdown",
                    options=[{'label': ticker, 'value': ticker} \
                             for ticker in self.tickers],
                    value=self.tickers[0]
                ),
                dcc.Graph(id="quote_ma_line"),
                dcc.Graph(id="ma_performance_line"),
                dcc.Graph(id="return_hist"),
                dcc.Checklist(
                    id='checklist_fundamentals',
                    options=[{'label': col, 'value': col} \
                             for col in self.fundamental_list],
                    value=[self.fundamental_list[0]],
                    labelStyle={'display': 'inline-block'}
                ),
                dcc.Graph(id='fundamentals_bar'),
                html.Label("Adjust Time Period for correlation matrix"),
                dcc.RangeSlider(
                    id="time_range_slider",
                    min=0,
                    max=len(date_range) - 1,
                    value=[0, len(date_range) - 1],
                    marks=marks,
                    step=1,
                    allowCross=False,
                ),
                dcc.Graph(id="corr_heatmap")
            ]
        )

    def _register_callbacks(self):
        """Functions defines the app callbacks to adjust
        the graphs basend on the given selections and filters.
        Here each callback and sub function is grouped by
        the defined callback options

        :return: None
        :rtype: None
        """
        @self.app.callback(
            [Output("quote_ma_line", "figure"),
             Output("ma_performance_line", "figure"),
             Output("return_hist", "figure")],
            [Input("tick_dropdown", "value")]
        )
        def _dropdwon_charts(tick_filter):
            """Function defines all graphs on
            which the ticker dropdown should be applied

            :param selected_ticker: Ticker from dropdown item
            :type selected_ticker: String
            :return: Line Chart and histogram
            :rtype: Plotly object
            """
            ma_data_filtered = self._tick_filter(
                df=self.ma_data,
                tick=tick_filter
            )
            quote_cols = [
                self.const_cols["quote"], 
                self.const_cols["sma1"], 
                self.const_cols["sma2"]
                ]
            performance_cols = [
                self.const_cols["position"], 
                self.const_cols["cumreturns"], 
                self.const_cols["cumstrategy"]
                ]
            ma_data_quote = ma_data_filtered[quote_cols]
            ma_data_performance = ma_data_filtered[performance_cols]
            ma_values_filtered = self.ma_values[tick_filter]
            performance = ma_values_filtered.loc[self.const_cols["performance"]]
            performance = np.round(performance, 2)
            returns_filtered = self.returns[tick_filter]
            quote_line_fig = {
                "data": [
                    {
                        "x": ma_data_quote.index, 
                        "y": ma_data_quote.loc[:, self.const_cols["quote"]], 
                        "type": "line", 
                        "name": "Quote",
                        "line": {"color": "blue"}
                    },
                    {
                        "x": ma_data_quote.index, 
                        "y": ma_data_quote.loc[:, self.const_cols["sma1"]], 
                        "type": "line", 
                        "name": f"SMA {int(ma_values_filtered.loc[self.const_cols["sma1"]])} Days",
                        "opacity": 0.75,
                        "line": {
                            "color": "green",
                            "width": 1.5
                        }
                    },
                    {
                        "x": ma_data_quote.index, 
                        "y": ma_data_quote.loc[:, self.const_cols["sma2"]], 
                        "type": "line", 
                        "name": f"SMA {int(ma_values_filtered.loc[self.const_cols["sma2"]])} Days",
                        "opacity": 0.75,
                        "line": {
                            "color": "red",
                            "width": 1.5
                        }
                    },
                ],
                "layout": {
                    "title": f"Optimal Moving Averages for ticker {tick_filter}",
                    "xaxis": {"title": "Date"},
                    "yaxis": {"title": "Values"},
                    "legend": {
                                'x': 0,
                                'y': 1,
                                'xanchor': 'left',
                                'yanchor': 'top'
                    }
                }
            }
            ma_performance_fig = {
                "data": [
                    {
                        'x': ma_data_performance.index, 
                        'y': ma_data_performance[self.const_cols["cumreturns"]], 
                        'mode': 'lines', 
                        'name': "Market Returns", 
                        'type': 'scatter',
                        "opacity": 0.5
                    },
                    {
                        'x': ma_data_performance.index, 
                        'y': ma_data_performance[self.const_cols["cumstrategy"]], 
                        'mode': 'lines', 
                        'name': "Strategy Returns", 
                        'type': 'scatter'
                    },
                    {
                        'x': ma_data_performance.index, 
                        'y': ma_data_performance[self.const_cols["position"]], 
                        'mode': 'lines', 
                        'name': 'Trading Strategy', 
                        'line': {'dash': 'dash'}, 
                        'yaxis': 'y2', 
                        'type': 'scatter'
                    }
                ],
                "layout": {
                        'title': f"Trading strategy with out-performance of {performance}",
                        'xaxis': {'title': 'Date'},
                        'yaxis': {'title': 'Cumulative Returns', 'side': 'right'},
                        'yaxis2': {
                            'title': 'Trading Strategy', 
                            'overlaying': 'y', 
                            'side': 'left', 
                            'showgrid': False},
                        "legend": {
                                'x': 0,
                                'y': 1,
                                'xanchor': 'left',
                                'yanchor': 'top'                            
                        }
                }

            }
            hist_fig = {
                "data": [
                    {
                        "x": returns_filtered, 
                        "type": "histogram",
                        "name": tick_filter
                    }
                ],
                "layout": {"title": f"Histogram for {tick_filter} returns"}
            }
            return quote_line_fig, ma_performance_fig, hist_fig
        
        @self.app.callback(
            Output("fundamentals_bar", "figure"),
            [Input("tick_dropdown", "value"),
            Input("checklist_fundamentals", "value")
             ]
        )
        def _dropdown_checklist_chart(tick_filter, fundamental_filter):
            """Function defines all graphs on which the checklist
            dropdown should be applied. Selecting columns triggers
            the callback and rearranges the calculated data

            :param tick_filter: _description_
            :type tick_filter: _type_
            :param fundamental_filter: _description_
            :type fundamental_filter: _type_
            :return: _description_
            :rtype: _type_
            """
            try:
                data = self._tick_filter(
                    df=self.fundamentals,
                    tick=tick_filter
                )
            except KeyError:
                data = pd.DataFrame(columns=self.fundamental_list)
                logging.warning(f"No fundamental data available for ticker {tick_filter}")

            data = data[fundamental_filter]
            fig = px.bar(data, 
                         barmode="group",
                         )
            return fig

        @self.app.callback(
            Output("corr_heatmap", "figure"),
            [Input("time_range_slider", "value")]
        )
        def _range_slider_charts(slider_array):
            """Function defines all graphs on which the time range slider 
            should be applied. Rearranging the time range triggers callback
            and recalculates the underlying data.

            :param slider_array: Containing lower and upper value for time range selection
            :type slider_value: Array
            :return: Correlation heatmap
            :rtype: Plotly object
            """
            date_range = self.returns.index
            start = date_range[slider_array[0]]
            end = date_range[slider_array[1]]

            start_filter = self.returns.index >= start
            end_filter = self.returns.index <= end
            returns_filtered = self.returns[(start_filter) & (end_filter)]
            returns_filtered = returns_filtered[self.tickers]
            corr_matrix = returns_filtered.corr()
            heatmap = px.imshow(corr_matrix, 
                                text_auto=True, 
                                aspect="auto", 
                                color_continuous_scale="RdBu_r"
                                )
            title = title=f"Return correlation {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}"
            heatmap.update_layout(title=title)
            return heatmap
        
    def run(self, debug=True):

        def run_dash():
            self.app.run_server(debug=debug, use_reloader=False)

        dash_thread = threading.Thread(target=run_dash)
        dash_thread.start()
        webbrowser.open_new("http://127.0.0.1:8050/")

class ModelBackTesting():
    def __init__(self, backtest_dict, validation_dict, model_list):
        self.backtest_dict = backtest_dict
        self.validation_dict = validation_dict
        self.model_list = model_list
        self.tickers = read_json("parameter.json")["ticker"]
        self.const_cols = read_json("constant.json")["columns"]
        self.app = Dash(__name__)
        self._setup_layout()
        self._register_callbacks()
        
    def _tick_filter(self, dict, tick):
        """Function filters dictionary for given ticker symbol.
        If symbol is not in ticker column or ticker column
        does not exist, the functin returns an empty dataframe

        :param dict: Prediction values to filter 
        for given ticker symbol
        :type dict: Dictionary
        :param tick: Ticker symbol for filtering
        :type tick: String
        :return: Filtered object
        :rtype: Dataframe
        """
        try:
            dict_filtered = dict[tick]
            return dict_filtered
        except KeyError:
            logging.error(f"Ticker {tick} not in model dictionary")
            return pd.DataFrame()
        
    def _setup_layout(self):
        """Function sets the layout for the dashboard app.
        All dashboard items like dropdowns, sliders and graphs 
        must be defined in this method
        """
        self.app.layout = html.Div(
            [   
                html.H1("Backtesting of forecast models"),
                dcc.Dropdown(
                    id="tick_dropdown",
                    options=[{'label': ticker, 'value': ticker} \
                             for ticker in self.tickers],
                    value=self.tickers[0]
                ),
                dcc.Checklist(
                    id='checklist_models',
                    options=[{'label': col, 'value': col} \
                             for col in self.model_list],
                    value=[self.model_list[0]],
                    labelStyle={'display': 'inline-block'}
                ),
                dcc.Graph(id="quote_backtest_line"),
                dash_table.DataTable(
                    id="validation_table",
                    columns=[{"name": i, "id": i} \
                             for i in self.validation_dict[self.tickers[0]].columns],
                    data=self.validation_dict[self.tickers[0]].to_dict('records')
                )
            ]
        )

    def _register_callbacks(self):
        """Functions defines the app callbacks to adjust
        the graphs basend on the given selections and filters.
        Here each callback and sub function is grouped by
        the defined callback options

        :return: None
        :rtype: None
        """
        @self.app.callback(
            Output("quote_backtest_line", "figure"),
            [Input("tick_dropdown", "value"), 
             Input("checklist_models", "value")]
        )
        def _dropdwon_checklist_charts(tick_filter, selected_models):
            backtest_data = self._tick_filter(
                dict=self.backtest_dict,
                tick=tick_filter
            )
            traces = []
            traces.append(
                go.Scatter(
                    x=backtest_data.index,
                    y=backtest_data[self.const_cols["quote"]],
                    mode="lines",
                    name=self.const_cols["quote"]
                )
            )
            for model in selected_models:
                traces.append(
                    go.Scatter(
                        x=backtest_data.index,
                        y=backtest_data[model],
                        mode="lines",
                        name=model
                    )
                )
            layout = go.Layout(
                title=f"Out-of-Sample Prediction for {tick_filter}",
                xaxis={"title": "Date"},
                yaxis={"title": "Value"},
                hovermode="closest"
            )
            backtest_fig = {
                "data": traces,
                "layout": layout
            }
            return backtest_fig
        
        @self.app.callback(
            [Output("validation_table", "columns"),
             Output("validation_table", "data")],
             Input("tick_dropdown", "value")
        )
        def _dropdown_table(tick_filter):
            validation_data = self._tick_filter(
                dict=self.validation_dict,
                tick=tick_filter
            )
            validation_data = validation_data.round(3)
            columns = [{"name": i, "id": i} \
                       for i in validation_data.columns]
            data = validation_data.to_dict('records')

            return columns, data

    def run(self, debug=True):

        def run_dash():
            self.app.run_server(debug=debug, use_reloader=False)

        dash_thread = threading.Thread(target=run_dash)
        dash_thread.start()
        webbrowser.open_new("http://127.0.0.1:8050/")
