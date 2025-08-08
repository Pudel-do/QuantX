from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
from core import logging_config
from misc.misc import *
from core.portfolio_generator import PortfolioGenerator
import pandas as pd
import numpy as np
import threading
import webbrowser
import logging

class DashboardAdapter:
    def __init__(
            self, ids, ticks,
            moving_avg, opt_moving_avg, 
            stock_rets, bench_rets, stock_infos, fundamentals,
            model_backtest, model_validation, models, actual_quotes
        ):
        self.ids = ids
        self.ticks = ticks
        self.moving_avg = moving_avg
        self.opt_moving_avg = opt_moving_avg
        self.stock_rets = stock_rets
        self.bench_rets = bench_rets
        self.stock_infos = stock_infos
        self.fundamentals = fundamentals
        self.model_backtest = model_backtest
        self.model_validation = model_validation
        self.models = models
        self.actual_quotes = actual_quotes
        self.const_cols = read_json("constant.json")["columns"]
        self.const_keys = read_json("constant.json")["keys"]
        self.port_types = list(self.const_keys.values())
        self.fundamental_cols = read_json("constant.json")["fundamentals"]["measures"]
        self.quote_marks, self.quote_date_range = self._init_time_range_values(self.moving_avg)
        self.rets_marks, self.rets_date_range = self._init_time_range_values(self.stock_rets)
        self.app = Dash(__name__)
        self._setup_layout()
        self._register_callbacks_analysis()
        self._register_callbacks_backtesting()
        self._register_callbacks_portfolio()
    
    def _init_time_range_values(self, ts):
        date_range = ts.index.unique()
        raw_marks = {i: str(date.year) \
            for i, date in enumerate(date_range)}
        seen_years = set()
        marks = {}
        for key, value in raw_marks.items():
            if value not in seen_years:
                marks[key] = value
                seen_years.add(value)
        return marks, date_range

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
        if not df.empty:
            try:
                filter_mask = df[tick_col] == tick
                df_filtered = df[filter_mask]
                df_filtered = df_filtered.drop(columns=tick_col)
                return df_filtered
            except KeyError:
                logging.error(f"Column {tick_col} or ticker symbol {tick} not in dataframe")
                return pd.DataFrame()
        else:
            return df

        
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
        
    def _filter_time_range(self, data, slider_array):
        date_range = data.index
        start = date_range[slider_array[0]]
        end = date_range[slider_array[1]]
        start_filter = data.index >= start
        end_filter = data.index <= end
        df_filtered = data[(start_filter) & (end_filter)]
        return df_filtered, start, end
    
    def _return_cleaning(self, df):
        df_clean = df.iloc[1:]
        df_clean = df_clean.fillna(0)
        return df_clean
    
    def _setup_layout(self):
        self.app.layout = html.Div(
        [   
            html.H1("Technical and fundamental stock analysis"),
            dcc.Dropdown(
                id="tick_dropdown_analysis",
                options=[{'label': ticker, 'value': ticker} \
                            for ticker in self.ids],
                value=self.ids[0]
            ),
            html.P(),
            dcc.RangeSlider(
                id="time_range_slider_quote",
                min=0,
                max=len(self.quote_date_range) - 1,
                value=[0, len(self.quote_date_range) - 1],
                marks=self.quote_marks,
                step=1,
                allowCross=False,
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
            dcc.Checklist(
                id='checklist_stock_infos',
                options=[{'label': col, 'value': col} \
                            for col in self.stock_infos.columns],
                value=[self.stock_infos.columns[0]],
                labelStyle={'display': 'inline-block'}
            ),
            dcc.Graph(id='stock_infos_bar'),
            html.Label("Adjust Time Period for correlation matrix"),
            dcc.RangeSlider(
                id="time_range_slider_corr_matrix",
                min=0,
                max=len(self.rets_date_range) - 1,
                value=[0, len(self.rets_date_range) - 1],
                marks=self.rets_marks,
                step=1,
                allowCross=False,
            ),
            dcc.Graph(id="corr_heatmap"),
            dcc.Dropdown(
                id="tick_dropdown_models",
                options=[{'label': ticker, 'value': ticker} \
                            for ticker in self.ids],
                value=self.ids[0]
            ),
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
            ),
            html.H1("Portfolio construction and performance"),
            dcc.Checklist(
                id='portfolio_checklist',
                options=[{'label': col, 'value': col} for col in self.port_types],
                value=[self.port_types[0]],
                inline=True
            ),
            html.P(),
            dcc.RangeSlider(
                id="time_range_slider_port",
                min=0,
                max=len(self.rets_date_range) - 1,
                value=[0, len(self.rets_date_range) - 1],
                marks=self.rets_marks,
                step=1,
                allowCross=False,
            ),
            dcc.Graph(id="portfolio_performances"),
            dash_table.DataTable(id="performance_table"),
            html.H3("Select portfolio for long positions"),
            dcc.Dropdown(
                id="portfolio_dropdown",
                options=[{'label': port_type, 'value': port_type} \
                            for port_type in self.port_types],
                value=self.port_types[0]
            ),
            html.P(),
            dash_table.DataTable(id="long_positions")
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
            [Input("tick_dropdown_analysis", "value"),
             Input("time_range_slider_quote", "value")]
        )
        def _dropdwon_charts(tick_filter, slider_array):
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
            ma_data_filtered, start, end = self._filter_time_range(
                data=ma_data_filtered,
                slider_array=slider_array
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
            quote = ma_data_quote.loc[:, self.const_cols["quote"]]
            sma1 = ma_data_quote.loc[:, self.const_cols["sma1"]]
            sma2 = ma_data_quote.loc[:, self.const_cols["sma2"]]

            rets = calculate_returns(quote)
            ann_mean_ret = rets.mean() * 252 * 100
            total_ret = (np.exp(rets.sum()) - 1) * 100

            quote_line_fig = {
                "data": [
                    {
                        "x": ma_data_quote.index, 
                        "y": quote, 
                        "type": "line", 
                        "name": "Quote",
                        "line": {"color": "blue"}
                    },
                    {
                        "x": ma_data_quote.index, 
                        "y": sma1, 
                        "type": "line", 
                        "name": f"SMA {int(ma_values_filtered.loc[self.const_cols["sma1"]])} Days",
                        "opacity": .75,
                        "line": {
                            "color": "green",
                            "width": 1
                        }
                    },
                    {
                        "x": ma_data_quote.index, 
                        "y": sma2, 
                        "type": "line", 
                        "name": f"SMA {int(ma_values_filtered.loc[self.const_cols["sma2"]])} Days",
                        "opacity": 0.75,
                        "line": {
                            "color": "red",
                            "width": 1
                        }
                    },
                ],
                "layout": {
                    "title": f"Annualized mean return of {ann_mean_ret: .2f}% and total return of {total_ret: .2f}% for period {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}",
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
            [Input("tick_dropdown_analysis", "value"),
            Input("checklist_fundamentals", "value")]
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

            data = self._filter_df(
                df=self.fundamentals,
                tick=tick_filter
            )
            if data.empty:
                data = pd.DataFrame(columns=self.fundamental_cols)
                logging.warning(f"No fundamental data available for ticker {tick_filter}")
            else:
                pass

            data = data[fundamental_filter]
            fig = px.bar(data, 
                         barmode="group",
                         )
            return fig
        @self.app.callback(
            Output("stock_infos_bar", "figure"),
            Input("checklist_stock_infos", "value")
        )
        def _dropdown_checklist_chart(stock_info_filter):
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
            data = self.stock_infos[stock_info_filter]
            fig = px.bar(data, 
                         barmode="group",
                         )
            return fig

        @self.app.callback(
            Output("corr_heatmap", "figure"),
            [Input("time_range_slider_corr_matrix", "value")]
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
            returns_filtered, start, end = self._filter_time_range(
                data=self.stock_rets,
                slider_array=slider_array
            )
            corr_matrix = returns_filtered.corr()
            heatmap = px.imshow(corr_matrix, 
                                text_auto=True, 
                                aspect="auto", 
                                color_continuous_scale="RdBu_r"
                                )
            title = f"Return correlation for period {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}"
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
            [Input("tick_dropdown_models", "value"), 
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
             Output("validation_table", "data"),
             Input("tick_dropdown_analysis", "value")
        )
        def _dropdown_table(tick_filter):
            validation_data = self._filter_dict(
                dict=self.model_validation,
                filter=tick_filter
            )
            validation_data = validation_data.round(3)
            validation_data.index.name = self.const_cols["measures"]
            validation_data.reset_index(inplace=True)
            data = validation_data.to_dict('records')
            return data
        
    def _register_callbacks_portfolio(self):
        """Functions defines the app callbacks to adjust
        the graphs basend on the given selections and filters.
        Here each callback and sub function is grouped by
        the defined callback options

        :return: None
        :rtype: None
        """

        future_rets = get_future_returns(
            tickers=self.ticks,
            rets=self.stock_rets
        )
        @self.app.callback(
            [Output('portfolio_performances', 'figure'),
             Output('performance_table', 'data'),
             Output("long_positions", "data")],
            [Input("time_range_slider_port", "value"),
             Input('portfolio_checklist', 'value'),
             Input("portfolio_dropdown", "value")]
        )
        def _checklist_charts(slider_array, selected_columns, port_filter):
            """Function defines all graphs on
            which the ticker dropdown should be applied

            :param selected_ticker: Ticker from dropdown item
            :type selected_ticker: String
            :return: Line Chart and histogram
            :rtype: Plotly object
            """
            hist_rets_filtered, start, end = self._filter_time_range(
                data=self.stock_rets,
                slider_array=slider_array
            )
            bench_rets_filtered, _, _ = self._filter_time_range(
                data=self.bench_rets,
                slider_array=slider_array
            )
            hist_rets_filtered = self._return_cleaning(hist_rets_filtered)
            bench_rets_filtered = self._return_cleaning(bench_rets_filtered)

            max_sharpe_weights = PortfolioGenerator(hist_rets_filtered).get_max_sharpe_weights()
            min_var_weights = PortfolioGenerator(hist_rets_filtered).get_min_var_weights()
            custom_weights = PortfolioGenerator(hist_rets_filtered).get_custom_weights()
            equal_weights = PortfolioGenerator(hist_rets_filtered).get_equal_weights()

            optimal_weights = {}
            optimal_weights[self.const_keys["MAX_SHARPE"]] = max_sharpe_weights
            optimal_weights[self.const_keys["MIN_VAR"]] = min_var_weights
            optimal_weights[self.const_keys["CUSTOM"]] = custom_weights
            optimal_weights[self.const_keys["EQUAL"]] = equal_weights

            hist_port_list = []
            future_post_list = []
            port_types = []
            for key, weights in optimal_weights.items():
                hist_port_rets = PortfolioGenerator(hist_rets_filtered).get_returns(weights)
                future_port_rets = PortfolioGenerator(future_rets).get_returns(weights)
                hist_port_rets.name = key
                future_port_rets.name = key
                hist_port_list.append(hist_port_rets)   
                future_post_list.append(future_port_rets)
                port_types.append(key)

            hist_port_rets = pd.concat(hist_port_list, axis=1)
            future_port_rets = pd.concat(future_post_list, axis=1)
            hist_idx = hist_port_rets.index
            future_idx = future_port_rets.index
            common_idx_mask = future_idx.isin(hist_idx)
            future_port_rets = future_port_rets[~common_idx_mask]
            port_rets = pd.concat(
                [hist_port_rets, future_port_rets],
                axis=0
            )
            cum_port_rets = cumulate_returns(port_rets)
            hist_port_mask = cum_port_rets.index.isin(hist_idx)
            cum_hist_port_rets = cum_port_rets[hist_port_mask]
            cum_future_port_rets = cum_port_rets[~hist_port_mask]
            cum_hist_bench_rets = cumulate_returns(bench_rets_filtered)
            bench_rets_filtered = bench_rets_filtered.squeeze()
            cum_hist_bench_rets = cum_hist_bench_rets.squeeze()

            port_performance = pd.DataFrame()
            for port_type, rets in hist_port_rets.items():
                ann_mean_ret, ann_mean_vol, sharpe_ratio, bench_corr = PortfolioGenerator(rets).get_portfolio_performance(bench_rets_filtered)
                ann_mean_ret = ann_mean_ret * 100
                port_performance.loc[port_type, self.const_cols["ann_mean_ret"]] = ann_mean_ret
                port_performance.loc[port_type, self.const_cols["ann_vola"]] = ann_mean_vol
                port_performance.loc[port_type, self.const_cols["sharpe_ratio"]] = sharpe_ratio
                port_performance.loc[port_type, self.const_cols["bench_corr"]] = bench_corr

            port_performance = port_performance.round(2)
            port_performance.index.name = self.const_cols["port_types"]
            port_performance.reset_index(inplace=True)
            table = port_performance.to_dict('records')
            fig = go.Figure()
            for col in selected_columns:
                fig.add_trace(go.Scatter(
                    x=cum_hist_port_rets.index,
                    y=cum_hist_port_rets[col],
                    mode="lines",
                    name=col,
                    line=dict(width=2, dash='solid')
                ))
                fig.add_trace(go.Scatter(
                    x=cum_future_port_rets.index,
                    y=cum_future_port_rets[col],
                    mode="lines",
                    name=col,
                    line=dict(width=2, dash='dash')
                ))
            fig.add_trace(go.Scatter(
                x=cum_hist_bench_rets.index,
                y=cum_hist_bench_rets,
                mode="lines",
                name=f"Benchmark {cum_hist_bench_rets.name}",
                line=dict(width=1, dash='solid')
            ))
            fig.update_layout(
                title=f"Portfolio performance for period {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}",
                xaxis_title="Date",
                yaxis_title="Cumulative returns",
                template="plotly"
            )
            actual_weights = {}
            actual_long_pos = {}
            for port_type, weights in optimal_weights.items():
                weight_dict, long_pos_dict = PortfolioGenerator(self.stock_rets).get_actual_invest(weights, self.actual_quotes)
                actual_weights[port_type] = weight_dict
                actual_long_pos[port_type] = long_pos_dict

            opt_dict_keys = list(optimal_weights.keys())
            act_dict_keys = list(actual_weights.keys())
            long_pos_keys = list(actual_long_pos.keys())
            common_keys = get_list_intersection(
                opt_dict_keys,
                act_dict_keys,
                long_pos_keys
            )
            result_dict = {}
            for type in common_keys:
                long_pos_results = pd.DataFrame(
                    index=self.ticks
                )
                long_pos_results.index.name = self.const_cols["company"]
                for tick in self.ticks:
                    opt_weight = optimal_weights[type].get(tick)
                    act_weight = actual_weights[type].get(tick)
                    n_shares = actual_long_pos[type].get(tick)[0]
                    invest = actual_long_pos[type].get(tick)[1]
                    long_pos_results.loc[tick, self.const_cols["opt_weight"]] = opt_weight
                    long_pos_results.loc[tick, self.const_cols["act_weight"]] = act_weight
                    long_pos_results.loc[tick, self.const_cols["long_pos"]] = n_shares
                    long_pos_results.loc[tick, self.const_cols["amount"]] = invest
                result_dict[type] = long_pos_results

            port_long_pos = self._filter_dict(
                dict=result_dict,
                filter=port_filter
            )
            port_long_pos = port_long_pos.round(2)
            port_long_pos.reset_index(inplace=True)
            data = port_long_pos.to_dict('records')
            return fig, table, data

    def run(self, debug=True):

        def run_dash():
            self.app.run_server(debug=debug, use_reloader=False)

        dash_thread = threading.Thread(target=run_dash)
        dash_thread.start()
        webbrowser.open_new("http://127.0.0.1:8050/")

