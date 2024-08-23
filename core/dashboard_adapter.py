from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
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
        ma_tickers = self.ma_data["Ticker"].unique()
        return_tickers = self.returns.columns
        tickers = list(set(ma_tickers) & set(return_tickers))
        date_range = self.returns.index
        self.app.layout = html.Div(
            [
                dcc.Dropdown(
                    id="Dropdown Ticker",
                    options=[{'label': ticker, 'value': ticker} \
                             for ticker in tickers],
                    value=tickers[0]
                ),
                dcc.Graph(id="LineChart"),
                dcc.Graph(id="Histogram"),
                html.Label("Adjust Time Period for correlation matrix"),
                dcc.RangeSlider(
                    id="Date Range Slider",
                    min=0,
                    max=len(date_range) - 1,
                    value=[0, len(date_range) - 1],
                    marks={i: date.strftime('%Y-%m') \
                           for i, date in enumerate(date_range) \
                            if date.month == 1},
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
                dcc.Graph(id="Heatmap")
            ]
        )

    def _register_callbacks(self):
        @self.app.callback(
            [Output("LineChart", "figure"),
             Output("Histogram", "figure")],
            [Input("Dropdown Ticker", "value")]
        )
        def update_dropdwon_charts(selected_ticker):
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
        
        @self.app.callback(
            Output('Heatmap', 'figure'),
            [Input('Date Range Slider', 'value')]
        )    
        def update_slider_charts(slider_value):
            # Determine the end date for the correlation period based on the slider value
            full_date_range = self.returns.index
            start_date = full_date_range[slider_value[0]]
            end_date = full_date_range[slider_value[1]]

            # Filter the returns DataFrame up to the selected date
            filtered_returns_df = self.returns[(self.returns.index >= start_date) & (self.returns.index <= end_date)]

            # Calculate the correlation matrix
            correlation_matrix = filtered_returns_df.corr()

            # Create the heatmap figure
            heatmap_figure = px.imshow(correlation_matrix, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
            heatmap_figure.update_layout(title=f"Return Correlation Heatmap ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})")

            return heatmap_figure
        
    def run(self, debug=True):

        def run_dash():
            self.app.run_server(debug=debug, use_reloader=False)

        dash_thread = threading.Thread(target=run_dash)
        dash_thread.start()
        webbrowser.open_new("http://127.0.0.1:8050/")