from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
from core import logging_config
from misc.misc import read_json
import pandas as pd
import numpy as np
import threading
import webbrowser
import logging

class DashboardAdapter:
    def __init__(
            self, ticks,
            moving_avg, opt_moving_avg, stock_rets, fundamentals, 
            model_backtest, model_validation, models, 
            cum_bench_rets, cum_hist_rets, cum_future_rets, 
            port_performance, long_pos, port_types, 
        ):
        self.ticks = ticks
        self.moving_avg = moving_avg
        self.opt_moving_avg = opt_moving_avg
        self.stock_rets = stock_rets
        self.fundamentals = fundamentals
        self.model_backtest = model_backtest
        self.model_validation = model_validation
        self.models = models
        self.cum_bench_rets = cum_bench_rets
        self.cum_hist_rets = cum_hist_rets
        self.cum_future_rets = cum_future_rets
        self.port_performance = port_performance
        self.long_pos = long_pos
        self.port_types = port_types
        self.const_cols = read_json("constant.json")["columns"]
        self.fundamental_cols = read_json("constant.json")["fundamentals"]["measures"]
        self.app = Dash(__name__)
        self._init_values()
        self._setup_layout()
        self._register_callbacks_analysis()
        self._register_callbacks_backtesting()
        self._register_callbacks_portfolio()
    
    def _init_values(self):
        date_range = self.stock_rets.index
        raw_marks = {i: str(date.year) \
            for i, date in enumerate(date_range)}
        seen_years = set()
        marks = {}
        for key, value in raw_marks.items():
            if value not in seen_years:
                marks[key] = value
                seen_years.add(value)
        self.marks = marks
        self.date_range = date_range

    def _filter_df(self, df, tick):
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
        
    def _filter_dict(self, dict, filter):
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
            dict_filtered = dict[filter]
            return dict_filtered
        except KeyError:
            logging.error(f"Ticker {filter} not in model dictionary")
            return pd.DataFrame()

    def _setup_layout(self):
        self.app.layout = html.Div(
        [   
            html.H1("Technical and fundamental stock analysis"),
            dcc.Dropdown(
                id="tick_dropdown",
                options=[{'label': ticker, 'value': ticker} \
                            for ticker in self.ticks],
                value=self.ticks[0]
            ),
            dcc.Graph(id="quote_ma_line"),
            dcc.Graph(id="ma_performance_line"),
            dcc.Graph(id="return_hist"),
            dcc.Checklist(
                id='checklist_fundamentals',
                options=[{'label': col, 'value': col} \
                            for col in self.fundamental_cols],
                value=[self.fundamental_cols[0]],
                labelStyle={'display': 'inline-block'}
            ),
            dcc.Graph(id='fundamentals_bar'),
            html.Label("Adjust Time Period for correlation matrix"),
            dcc.RangeSlider(
                id="time_range_slider",
                min=0,
                max=len(self.date_range) - 1,
                value=[0, len(self.date_range) - 1],
                marks=self.marks,
                step=1,
                allowCross=False,
            ),
            dcc.Graph(id="corr_heatmap"),
            html.H1("Backtesting of forecast models"),
            dcc.Checklist(
                id='checklist_models',
                options=[{'label': col, 'value': col} \
                            for col in self.models],
                value=[self.models[0]],
                labelStyle={'display': 'inline-block'}
            ),
            dcc.Graph(id="quote_backtest_line"),
            dash_table.DataTable(
                id="validation_table",
                columns=[{"name": i, "id": i} \
                            for i in self.model_validation[self.ticks[0]].columns],
                data=self.model_validation[self.ticks[0]].to_dict('records')
            ),
            html.H1("Portfolio construction and performance"),
            dcc.Checklist(
                id='portfolio_checklist',
                options=[{'label': col, 'value': col} for col in self.port_types],
                value=[self.port_types[0]],
                inline=True
            ),
            dcc.Graph(id="portfolio_performances"),
            dash_table.DataTable(
                id="performance_table",
                data=self.port_performance.to_dict('records')
            ),
            html.H3("Select portfolio for long positions"),
            dcc.Dropdown(
                id="portfolio_dropdown",
                options=[{'label': port_type, 'value': port_type} \
                            for port_type in self.port_types],
                value=self.port_types[0]
            ),
            html.P(),
            dash_table.DataTable(
                id="long_positions",
                columns=[{"name": i, "id": i} \
                            for i in self.long_pos[self.port_types[0]].columns],
                data=self.long_pos[self.port_types[0]].to_dict('records')
            )
        ]
    )

    def _register_callbacks_analysis(self):
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
            ma_data_filtered = self._filter_df(
                df=self.moving_avg,
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
            ma_values_filtered = self.opt_moving_avg[tick_filter]
            performance = ma_values_filtered.loc[self.const_cols["performance"]]
            performance = np.round(performance, 3)
            returns_filtered = self.stock_rets[tick_filter]
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
                    "title": f"Optimal Moving Averages for {tick_filter}",
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
                data = self._filter_df(
                    df=self.fundamentals,
                    tick=tick_filter
                )
            except KeyError:
                data = pd.DataFrame(columns=self.fundamental_cols)
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
            date_range = self.stock_rets.index
            start = date_range[slider_array[0]]
            end = date_range[slider_array[1]]

            start_filter = self.stock_rets.index >= start
            end_filter = self.stock_rets.index <= end
            returns_filtered = self.stock_rets[(start_filter) & (end_filter)]
            returns_filtered = returns_filtered[self.ticks]
            corr_matrix = returns_filtered.corr()
            heatmap = px.imshow(corr_matrix, 
                                text_auto=True, 
                                aspect="auto", 
                                color_continuous_scale="RdBu_r"
                                )
            title = title=f"Return correlation {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}"
            heatmap.update_layout(title=title)
            return heatmap
        
    def _register_callbacks_backtesting(self):
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
            backtest_data = self._filter_dict(
                dict=self.model_backtest,
                filter=tick_filter
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
            validation_data = self._filter_dict(
                dict=self.model_validation,
                filter=tick_filter
            )
            validation_data = validation_data.round(3)
            columns = [{"name": i, "id": i} \
                       for i in validation_data.columns]
            data = validation_data.to_dict('records')

            return columns, data
        
    def _register_callbacks_portfolio(self):
        """Functions defines the app callbacks to adjust
        the graphs basend on the given selections and filters.
        Here each callback and sub function is grouped by
        the defined callback options

        :return: None
        :rtype: None
        """
        @self.app.callback(
            Output('portfolio_performances', 'figure'),
            Input('portfolio_checklist', 'value')
        )
        def _checklist_charts(selected_columns):
            """Function defines all graphs on
            which the ticker dropdown should be applied

            :param selected_ticker: Ticker from dropdown item
            :type selected_ticker: String
            :return: Line Chart and histogram
            :rtype: Plotly object
            """
            bench_rets = self.cum_bench_rets.squeeze()
            fig = go.Figure()
            for col in selected_columns:
                fig.add_trace(go.Scatter(
                    x=self.cum_hist_rets.index,
                    y=self.cum_hist_rets[col],
                    mode="lines",
                    name=col,
                    line=dict(width=2, dash='solid')
                ))
                fig.add_trace(go.Scatter(
                    x=self.cum_future_rets.index,
                    y=self.cum_future_rets[col],
                    mode="lines",
                    name=col,
                    line=dict(width=2, dash='dash')
                ))
            fig.add_trace(go.Scatter(
                x=bench_rets.index,
                y=bench_rets,
                mode="lines",
                name=f"Benchmark {bench_rets.name}",
                line=dict(width=1, dash='solid')
            ))
            fig.update_layout(
                title="Portfolio performance for different portfolio types",
                xaxis_title="Date",
                yaxis_title="Cumulative returns",
                template="plotly"
            )
            return fig
        
        @self.app.callback(
            [Output("long_positions", "columns"),
            Output("long_positions", "data")],
            Input("portfolio_dropdown", "value")
        )
        def _dropdown_table(port_filter):
            port_long_pos = self._filter_dict(
                dict=self.long_pos,
                filter=port_filter
            )
            columns = [{"name": i, "id": i} \
                       for i in port_long_pos.columns]
            data = port_long_pos.to_dict('records')
            return columns, data

    def run(self, debug=True):

        def run_dash():
            self.app.run_server(debug=debug, use_reloader=False)

        dash_thread = threading.Thread(target=run_dash)
        dash_thread.start()
        webbrowser.open_new("http://127.0.0.1:8050/")
