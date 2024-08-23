from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import time
import threading
import webbrowser
import logging
import os

class AnalysisDashboard:
    def __init__(self, tickers, ma_data, returns):
        self.tickers = tickers
        self.ma_data = ma_data
        self.returns = returns
        self.app = Dash(__name__)
        self._setup_layout()
        self._register_callbacks()

    def _setup_layout(self):
        self.app.layout = html.Div(
            [
                dcc.Dropdown(
                    id="Dropdown Ticker",
                    options=[{'label': ticker, 'value': ticker} \
                             for ticker in self.tickers],
                    value=self.tickers[0]
                ),
                dcc.Graph(id="LineChart"),
                dcc.Graph(id="Histogram")
            ]
        )

    def _register_callbacks(self):
        @self.app.callback(
            [Output("LineChart", "figure"),
             Output("Histogram", "figure")],
            [Input("Dropdown Ticker", "value")]
        )
        def update_charts(selected_ticker):
            filter_mask_line = self.ma_data["Ticker"] == selected_ticker
            filtered_line = self.ma_data[filter_mask_line]
            filtered_hist = self.returns[selected_ticker]
            line_chart_fig = {
                'data': [
                    {
                        'x': filtered_line.index, 
                        'y': filtered_line["Quote"], 
                        'type': 'line', 
                        'name': 'Quote',
                        'line': {'color': 'blue'}
                    },
                    {
                        'x': filtered_line.index, 
                        'y': filtered_line["MovingAverage42"], 
                        'type': 'line', 
                        'name': 'Moving Average Short',
                        'line': {'color': 'green'}
                    },
                    {
                        'x': filtered_line.index, 
                        'y': filtered_line["MovingAverage252"], 
                        'type': 'line', 
                        'name': 'Moving Average Long',
                        'line': {'color': 'red'}
                    },
                ],
                'layout': {
                    'title': f'Stock Data for {selected_ticker}',
                    'xaxis': {'title': 'Date'},
                    'yaxis': {'title': 'Values'},
                }
            }
            hist_fig = {
                'data': [
                    {
                        'x': filtered_hist, 
                        'type': 'histogram', 
                        'name': selected_ticker
                    }
                ],
                'layout': {'title': f'Histogram for {selected_ticker} returns'}
            }
            return line_chart_fig, hist_fig
        
    def run(self, debug=True):

        def run_dash():
            self.app.run_server(debug=debug, use_reloader=False)

        dash_thread = threading.Thread(target=run_dash)
        dash_thread.start()
        webbrowser.open_new("http://127.0.0.1:8050/")