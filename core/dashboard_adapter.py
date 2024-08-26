from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from core import logging_config
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import threading
import webbrowser
import logging
import os

class AnalysisDashboard:
    def __init__(self, 
                 tickers, 
                 ma_data, 
                 returns, 
                 fundamentals, 
                 fundamental_list):
        """

        :param tickers: Tickers from project inputs.json
        :type tickers: List
        :param ma_data: Dictionary with moving average values for each stock
        :type ma_data: Dictionary
        :param returns: Log returns for each stock
        :type returns: Dataframe
        :param fundamentals: Fundamental data for each stock
        :type fundamentals: Dictionary
        """
        self.tickers = tickers
        self.ma_data = ma_data
        self.returns = returns
        self.fundamentals = fundamentals
        self.fundamental_list = fundamental_list
        self.app = Dash(__name__)
        self._setup_layout()
        self._register_callbacks()

        log = logging.getLogger('dash')
        log.setLevel(logging.ERROR)

    def _setup_layout(self):
        #TODO: Outsource preprocessing in separate function
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
                dcc.Graph(id="ma_line"),
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
            [Output("ma_line", "figure"),
             Output("return_hist", "figure")],
            [Input("tick_dropdown", "value")]
        )
        def dropdwon_charts(tick_filter):
            """Function defines all graphs on
            which the ticker dropdown should be applied

            :param selected_ticker: Ticker from dropdown item
            :type selected_ticker: String
            :return: Line Chart and histogram
            :rtype: Plotly object
            """
            ma_line_filtered = self.ma_data[tick_filter]
            returns_filtered = self.returns[tick_filter]
            line_chart_fig = {
                'data': [
                    {
                        "x": ma_line_filtered.index, 
                        "y": ma_line_filtered.iloc[:, 0], 
                        "type": "line", 
                        "name": "Quote",
                        "line": {"color": "blue"}
                    },
                    {
                        "x": ma_line_filtered.index, 
                        "y": ma_line_filtered.iloc[:, 1], 
                        "type": "line", 
                        "name": f"{ma_line_filtered.iloc[:, 1].name}",
                        "line": {"color": "green"}
                    },
                    {
                        "x": ma_line_filtered.index, 
                        "y": ma_line_filtered.iloc[:, 2], 
                        "type": "line", 
                        "name": f"{ma_line_filtered.iloc[:, 2].name}",
                        "line": {"color": "red"}
                    },
                ],
                "layout": {
                    "title": f"Moving Averages for {tick_filter}",
                    "xaxis": {"title": "Date"},
                    "yaxis": {"title": "Values"},
                }
            }
            hist_fig = {
                "data": [
                    {
                        "x": returns_filtered, 
                        "type": "histogram",
                        "nbins": 100,
                        "name": tick_filter
                    }
                ],
                "layout": {"title": f"Histogram for {tick_filter} returns"}
            }
            return line_chart_fig, hist_fig
        
        @self.app.callback(
            Output("fundamentals_bar", "figure"),
            [Input("tick_dropdown", "value"),
            Input("checklist_fundamentals", "value")
             ]
        )
        def dropdown_checklist_chart(tick_filter, fundamental_filter):
            try:
                data = self.fundamentals[tick_filter]
                data = data[fundamental_filter]
            except KeyError:
                logging.warning(f"No fundamental data for ticker {tick_filter}")
                data = pd.DataFrame()
            fig = px.bar(data, 
                         barmode="group"
                         )
            return fig

        @self.app.callback(
            Output("corr_heatmap", "figure"),
            [Input("time_range_slider", "value")]
        )
        def range_slider_charts(slider_array):
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