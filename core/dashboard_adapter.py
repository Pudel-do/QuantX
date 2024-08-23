from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import time
import threading
import webbrowser
import logging
import os

class AnalysisDashboard:
    def __init__(self, quotes):
        self.quotes = quotes
        self.app = Dash(__name__)
        self.setup_layout()
        self.register_callbacks()

    def layout_with_dropdown(self, df, chart_id):
        tickers = df['Ticker'].unique()
        return html.Div([
            dcc.Dropdown(
                id=f'Ticker {chart_id}',
                options=[{'label': ticker, 'value': ticker} for ticker in tickers],
                value=tickers[0],
                clearable=False,
                style={'width': '50%'}
            ),
            dcc.Graph(id=chart_id)
        ])

    def setup_layout(self):
        self.app.layout = html.Div([
        self.layout_with_dropdown(self.quotes, 'Moving Averages'),
        self.layout_with_dropdown(self.quotes, 'Return Histogram')
        ])

    def register_callbacks(self):
        # Register callbacks for the line chart
        @self.app.callback(
            Output('Moving Averages', 'figure'),
            [Input('Ticker Moving Averages', 'value')]
        )
        def update_line_chart(selected_ticker):
            filter_mask = self.quotes["Ticker"] == selected_ticker
            filtered_df = self.quotes[filter_mask]
            fig = {
                'data': [
                    {
                        'x': filtered_df.index, 
                        'y': filtered_df['Quote'], 
                        'type': 'line', 
                        'name': 'Quote',
                        'line': {'color': 'blue'}
                    },
                    {
                        'x': filtered_df.index, 
                        'y': filtered_df['MA_42'], 
                        'type': 'line', 
                        'name': 'SMA1',
                        'line': {'color': 'green'}
                    },
                    {
                        'x': filtered_df.index, 
                        'y': filtered_df['MA_252'], 
                        'type': 'line', 
                        'name': 'SMA2',
                        'line': {'color': 'red'}
                    },
                ],
                'layout': {
                    'title': f'Stock Data for {selected_ticker}',
                    'xaxis': {'title': 'Date'},
                    'yaxis': {'title': 'Values'},
                }
            }
            return fig

        # Register callbacks for the histogram
        @self.app.callback(
            Output('Return Histogram', 'figure'),
            [Input('Ticker Return Histogram', 'value')]
        )
        def update_histogram(selected_ticker):
            filter_mask = self.quotes["Ticker"] == selected_ticker
            filtered_df = self.quotes[filter_mask]
            return {
                'data': [{'x': filtered_df['Quote'], 'type': 'histogram', 'name': selected_ticker}],
                'layout': {'title': f'Histogram for {selected_ticker}'}
            }
        
    def run(self, debug=True):
        def open_browser():
            webbrowser.open_new("http://127.0.0.1:8050/")

        def run_dash():
            self.app.run_server(debug=debug, use_reloader=False)
        
        # Start the Dash app in a separate thread
        dash_thread = threading.Thread(target=run_dash)
        dash_thread.start()

        # Open the web browser
        open_browser()