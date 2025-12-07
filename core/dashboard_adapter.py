from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output, State, MATCH, ALL
import plotly.express as px
import plotly.graph_objects as go
from core import logging_config
from misc.utils import *
from core.portfolio_generator import PortfolioGenerator
import pandas as pd
import numpy as np
import threading
import webbrowser
import logging

class DashboardAdapter:
    def __init__(
            self, assets, ticks, tick_mapping,
            moving_avg, opt_moving_avg, port_types,
            stock_rets, bench_rets, stock_infos, fundamentals,
            model_backtest, model_validation, models, model_data, 
            actual_quotes
        ):
        self.app = Dash(__name__)
        self.assets = assets
        self.ticks = ticks
        self.tick_mapping = tick_mapping
        self.models = models
        self.model_data = model_data
        self.params = read_json("parameter.json")
        self.const_cols = read_json("constant.json")["columns"]
        self.fundamental_cols = read_json("constant.json")["fundamentals"]["measures"]
        self.weight_list = [self.const_cols["opt_weight"], self.const_cols["act_weight"]]
        self.port_types = port_types
        self.moving_avg = rename_dataframe(df=moving_avg, tick_map=tick_mapping)
        self.opt_moving_avg = rename_dataframe(df=opt_moving_avg, tick_map=tick_mapping)
        self.stock_rets = rename_dataframe(df=stock_rets, tick_map=tick_mapping)
        self.bench_rets = rename_dataframe(df=bench_rets, tick_map=tick_mapping)
        self.stock_infos = rename_dataframe(df=stock_infos, tick_map=tick_mapping)
        self.fundamentals = rename_dataframe(df=fundamentals, tick_map=tick_mapping)
        self.model_backtest = rename_dictionary(dict=model_backtest, tick_map=tick_mapping)
        self.model_validation = rename_dictionary(dict=model_validation, tick_map=tick_mapping)
        self.actual_quotes = rename_dictionary(dict=actual_quotes, tick_map=tick_mapping)
        self.quote_marks, self.quote_date_range = self._init_time_range_values(self.moving_avg)
        self.rets_marks, self.rets_date_range = self._init_time_range_values(self.stock_rets)
        self._setup_layout()
        self._register_callbacks_analysis()
        self._register_callbacks_backtesting()
        self._register_callbacks_portfolio()
    
    def _setup_layout(self):
        self.app.layout = html.Div(
        [   
            html.H1("Technical analysis"),
            html.Label("Adjust time period for return analysis"),
            html.P(),
            dcc.RangeSlider(
                id="time_range_slider_returns",
                min=0,
                max=len(self.rets_date_range) - 1,
                value=[0, len(self.rets_date_range) - 1],
                marks=self.rets_marks,
                step=1,
                allowCross=False,
            ),
            dcc.Graph(id="cumulated_stock_returns"),
            dash_table.DataTable(id="stock_performance_table"),
            dcc.Graph(id="corr_heatmap"),
            html.P(),
            html.H1("Technical and fundamental analysis for selected asset"),
            dcc.Dropdown(
                id="tick_dropdown_analysis",
                options=[{'label': asset, 'value': asset} \
                            for asset in self.assets],
                value=self.assets[0]
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
                value=[self.params["stock_infos"][0]],
                labelStyle={'display': 'inline-block'}
            ),
            dcc.Graph(id='stock_infos_bar'),
            dcc.Dropdown(
                id="tick_dropdown_models",
                options=[{'label': asset, 'value': asset} \
                            for asset in self.assets],
                value=self.assets[0]
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
            html.H1("Portfolio analysis"),
            html.P(),
            dcc.Dropdown(
                id="weight_filter",
                options=[{'label': weight_type, 'value': weight_type} \
                            for weight_type in self.weight_list],
                value=self.const_cols["opt_weight"]
            ),
            html.P(),
            dcc.Checklist(
                id='portfolio_constituents',
                options=[{'label': asset, 'value': asset} \
                            for asset in self.assets],
                value=self.assets,
                inline=True
            ),
            html.P(),
            html.Div(id="weights_container"),
            html.Div(id="weights_validation_message", style={"color": "red", "font-weight": "bold"}),
            dcc.Store(id="custom_weights_store", data={}),
            dcc.Store(id="previous_constituents_store", data=[]),
            html.P(),
            html.P(),
            dcc.Checklist(
                id='portfolio_checklist',
                options=[{'label': col, 'value': col} \
                         for col in list(self.port_types.values())],
                value=[list(self.port_types.values())[0]],
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
                            for port_type in list(self.port_types.values())],
                value=list(self.port_types.values())[0]
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
            ann_mean_ret = calc_annualized_mean_return(rets) * 100
            total_ret = calc_total_return(rets) * 100
            quote_plot = quote.fillna(method="ffill")

            quote_line_fig = {
                "data": [
                    {
                        "x": ma_data_quote.index, 
                        "y": quote_plot, 
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
                    "title": f"Annualized mean return for {tick_filter} of {ann_mean_ret: .2f}% and total return of {total_ret: .2f}% for period {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}",
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
                logging.warning(f"No fundamental data available for company {tick_filter}")
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
            [
            Output("cumulated_stock_returns", "figure"),
            Output("stock_performance_table", "data"),
            Output("corr_heatmap", "figure")
            ],
            Input("time_range_slider_returns", "value")
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

            cum_returns = cumulate_returns(returns=returns_filtered)
            cum_returns = cum_returns.fillna(method="ffill")
            cum_returns_fig = px.line(
                cum_returns, 
                x=cum_returns.index, 
                y=cum_returns.columns,
                title=f"Cumulative stock returns for period {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}",
                labels={"value": "Cumulative Returns", "variable": self.const_cols["asset"]}
            )

            corr_matrix = returns_filtered.corr()
            corr_heatmap = px.imshow(corr_matrix, 
                                text_auto=True, 
                                aspect="auto", 
                                color_continuous_scale="RdBu_r"
                                )
            title = f"Return correlation for period {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}"
            corr_heatmap.update_layout(title=title)

            performance_table = pd.DataFrame(index=self.assets)
            for col, values in returns_filtered.items():
                total_ret = calc_total_return(values) * 100
                ann_mean_ret = calc_annualized_mean_return(values) * 100
                ann_vola = calc_annualized_vola(values)
                performance_table.loc[col, self.const_cols["total_ret"]] = total_ret
                performance_table.loc[col, self.const_cols["ann_mean_ret"]] = ann_mean_ret
                performance_table.loc[col, self.const_cols["ann_vola"]] = ann_vola

            performance_table = performance_table.round(2)
            performance_table.index.name = self.const_cols["asset"]
            performance_table.reset_index(inplace=True)
            performance_table = performance_table.to_dict('records')

            return cum_returns_fig, performance_table, corr_heatmap
        
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
            try:
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
            except:
                traces = []
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
             Input("tick_dropdown_models", "value")
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
    
        def distribute_delta_additive(weights: dict, tickers: list) -> dict:
            """
            Take weights (for tickers subset) and add delta = 1 - sum(weights)
            equally to each ticker (delta_per = delta / n). Return new dict.
            If sum(weights) == 0 -> keep zeros (do not auto-equalize).
            """
            n = len(tickers)
            if n == 0:
                return {}

            s = sum(weights.values())
            delta = 1.0 - s

            # If sum is zero -> keep zeros (user hasn't entered anything).
            if abs(s) < 1e-12:
                return {t: 0.0 for t in tickers}

            delta_per = delta / n
            updated = {t: float(weights.get(t, 0.0)) + delta_per for t in tickers}

            # if any negative, clamp to zero and renormalize additively again
            if any(v < 0 for v in updated.values()):
                for k in list(updated.keys()):
                    if updated[k] < 0:
                        updated[k] = 0.0
                s2 = sum(updated.values())
                if s2 > 0:
                    delta2 = 1.0 - s2
                    delta2_per = delta2 / n
                    updated = {t: float(updated[t]) + delta2_per for t in tickers}
            return updated

        # --------------------
        # 1) Render inputs from store (single writer to weights_container)
        # --------------------
        @self.app.callback(
            Output("weights_container", "children"),
            Input("portfolio_constituents", "value"),
            State("custom_weights_store", "data"),
        )
        def render_weight_inputs(constituents, stored_weights):
            stored_weights = stored_weights or {}
            if not constituents:
                return []

            children = []
            for tick in constituents:
                raw_val = stored_weights.get(tick, 0.0)
                # show with two decimals in UI but do NOT change stored precision
                display_val = f"{raw_val:.2f}"
                children.append(
                    html.Div([
                        html.Label(f"{tick} weight:"),
                        dcc.Input(
                            id={"type": "weight_input", "ticker": tick},
                            type="number",
                            min=0,
                            max=1,
                            step=0.01,
                            value=float(display_val),
                            debounce=True,
                            style={"width": "110px", "marginLeft": "6px"}
                        )
                    ], style={"marginBottom": "6px", "display": "flex", "alignItems": "center"})
                )
            return children

        # --------------------
        # 2) Update store when inputs change OR constituents change.
        #    - keeps previously stored values for tickers that remain
        #    - adds new tickers with value 0.0
        #    - when constituents reduced -> compute additive delta and distribute equally
        # --------------------
        @self.app.callback(
            Output("previous_constituents_store", "data"),
            Input("portfolio_constituents", "value")
        )
        def update_previous_constituents(new_constituents):
            return new_constituents or []


        # ----------------------------------------------------------
        # Update custom_weights_store with correct removal detection
        # ----------------------------------------------------------
        @self.app.callback(
            Output("custom_weights_store", "data"),
            Input("portfolio_constituents", "value"),
            Input({"type": "weight_input", "ticker": ALL}, "value"),
            State({"type": "weight_input", "ticker": ALL}, "id"),
            State("custom_weights_store", "data"),
        )
        def update_custom_weights_store(new_constituents, input_values, input_ids, stored_weights):
            """
            Robust update of stored custom weights.

            Key improvements:
            - Use `stored_weights` (State) as the *previous* snapshot to detect removed tickers
            and compute the exact weight that was removed.
            - Apply any fresh UI inputs on top of the snapshot.
            - If tickers were removed, distribute the exact removed weight additively across
            remaining tickers (equal share).
            - If no removal, just persist raw values (new tickers get 0.0).
            """

            # normalize inputs
            stored_prev = (stored_weights or {}).copy()   # previous snapshot (before this call)
            new_constituents = new_constituents or []

            # 1) Apply recent UI input values onto a working copy (but keep stored_prev for removal calc)
            working = stored_prev.copy()
            if input_values and input_ids:
                for item, val in zip(input_ids, input_values):
                    ticker = item.get("ticker")
                    if ticker is None:
                        continue
                    try:
                        working[ticker] = float(val) if (val is not None) else 0.0
                    except Exception:
                        working[ticker] = 0.0

            # 2) If no constituents selected -> clear
            if not new_constituents:
                return {}

            # 3) Detect removed tickers by comparing previous snapshot keys with new constituents
            prev_keys = set(stored_prev.keys())
            new_keys = set(new_constituents)
            removed = prev_keys - new_keys

            # 4) Build filtered dict for the currently active constituents using the working copy
            filtered = {t: float(working.get(t, 0.0)) for t in new_constituents}

            # 5) If removal detected -> compute removed weight from the previous snapshot
            if removed:
                # compute weight that belonged to removed tickers in the *previous* snapshot
                removed_weight = sum(stored_prev.get(r, 0.0) for r in removed)

                # sum of remaining weights in previous snapshot (for info / sanity)
                remaining_prev_sum = sum(stored_prev.get(t, 0.0) for t in new_constituents)

                # If previous snapshot had all zeros (user never entered values), keep zeros
                if abs(remaining_prev_sum) < 1e-12 and abs(removed_weight) < 1e-12:
                    # fallback: keep zeros (user hasn't entered anything yet)
                    return {t: 0.0 for t in new_constituents}

                # Distribute removed_weight additively equally among remaining tickers
                n = len(new_constituents)
                delta_per = removed_weight / n

                adjusted = {t: float(filtered.get(t, 0.0)) + delta_per for t in new_constituents}

                # guard: clamp negatives (shouldn't happen with additive distribution)
                for k in adjusted:
                    if adjusted[k] < 0:
                        adjusted[k] = 0.0

                # final normalization to remove tiny float errors -> force sum == 1 if possible
                s = sum(adjusted.values())
                if s <= 1e-12:
                    # fallback equal weights
                    eq = 1.0 / n
                    return {t: eq for t in new_constituents}
                else:
                    # normalize to exact sum 1 to avoid accumulation of floating point drift
                    normalized = {t: adjusted[t] / s for t in new_constituents}
                    return normalized

            # 6) No removal -> just return the working dict (persist raw / partial values)
            # Ensure all constituents are present in the result (new ones get 0.0)
            result = {t: float(working.get(t, 0.0)) for t in new_constituents}
            return result

        # --------------------
        # 3) Validation message callback (always up-to-date, listens to store)
        # --------------------
        @self.app.callback(
            Output("weights_validation_message", "children"),
            Output("weights_validation_message", "style"),
            Input("custom_weights_store", "data"),
            Input("portfolio_constituents", "value"),
        )
        def validate_custom_weights_display(store_data, constituents):
            store_data = store_data or {}
            constituents = constituents or []

            if not constituents:
                return "", {"display": "none"}

            # Validation uses stored (raw or adjusted) weights
            filtered = {t: float(store_data.get(t, 0.0)) for t in constituents}
            s = sum(filtered.values())
            deviation = 1.0 - s

            tol = 0.001
            if abs(deviation) <= tol:
                msg = f"✔️ Custom Weights gültig. Summe = {s:.2f}"
                style = {"color": "green", "fontWeight": "bold"}
            else:
                # If user hasn't entered any values (s very small), be explicit
                if s < 1e-9:
                    msg = f"⚠️ Keine Custom Weights eingegeben. Summe = {s:.2f}"
                else:
                    msg = f"⚠️ Summe = {s:.2f}. Abweichung zu 1: {deviation:+.2f}"
                style = {"color": "red", "fontWeight": "bold"}

            return msg, style

        # --------------------
        # Prepare future returns once (useful for portfolio calculations)
        # --------------------
        future_rets = get_future_returns(
            tickers=self.ticks,
            rets=self.stock_rets,
            model_data=self.model_data
        )
        future_rets = rename_dataframe(future_rets, tick_map=self.tick_mapping)

        # --------------------
        # 4) Portfolio callback: listens to store and re-runs when weights change
        #    Note: weights come from store (already additive-adjusted)
        # --------------------
        @self.app.callback(
            [
                Output('portfolio_performances', 'figure'),
                Output('performance_table', 'data'),
                Output("long_positions", "data")
            ],
            [
                Input("weight_filter", "value"),
                Input("portfolio_constituents", "value"),
                Input("time_range_slider_port", "value"),
                Input('portfolio_checklist', 'value'),
                Input("portfolio_dropdown", "value"),
                Input("custom_weights_store", "data"),  # triggers when weights updated
            ]
        )
        def update_portfolio(
            weight_filter,
            constituents,
            slider_array,
            selected_port_types,
            longpos_port_type,
            custom_weights_store
        ):

            # guard
            if not constituents:
                return go.Figure(), [], []

            custom_weights_store = custom_weights_store or {}
            # take weights for selected constituents
            custom_weights = {tick: float(custom_weights_store.get(tick, 0.0)) for tick in constituents}

            hist_rets, start, end = self._filter_time_range(
                data=self.stock_rets,
                slider_array=slider_array
            )
            bench_rets, _, _ = self._filter_time_range(
                data=self.bench_rets,
                slider_array=slider_array
            )

            hist_rets = self._return_cleaning(hist_rets, constituents)
            bench_rets = self._return_cleaning(bench_rets, constituents)
            future_rets_filtered = future_rets[constituents]

            pg = PortfolioGenerator(hist_rets)
            weights = {
                self.port_types["max_sharpe"]: pg.get_max_sharpe_weights(),
                self.port_types["min_var"]: pg.get_min_var_weights(),
                self.port_types["equal"]: pg.get_equal_weights()
            }

            if self.params["use_custom_weights"]:
                weights[self.port_types["custom"]] = custom_weights

            weights_filtered = {
                k: v for k, v in weights.items() if k in selected_port_types
            }

            hist_list = []
            future_list = []

            for port_type, wdict in weights_filtered.items():
                hist_port = PortfolioGenerator(hist_rets).get_returns(wdict)
                fut_port = PortfolioGenerator(future_rets_filtered).get_returns(wdict)
                hist_port.name = port_type
                fut_port.name = port_type
                hist_list.append(hist_port)
                future_list.append(fut_port)

            hist_df = pd.concat(hist_list, axis=1)
            future_df = pd.concat(future_list, axis=1)

            future_df = future_df[~future_df.index.isin(hist_df.index)]

            port_rets = pd.concat([hist_df, future_df], axis=0)
            cum_rets = cumulate_returns(port_rets)
            cum_hist = cum_rets.loc[hist_df.index]
            cum_fut = cum_rets.loc[future_df.index]
            bench_cum = cumulate_returns(bench_rets).squeeze()

            performance = pd.DataFrame()
            for col in hist_df.columns:
                ann_ret, ann_vol, sharpe, corr = \
                    PortfolioGenerator(hist_df[col]).get_portfolio_performance(bench_rets.squeeze())

                performance.loc[col, self.const_cols["ann_mean_ret"]] = ann_ret * 100
                performance.loc[col, self.const_cols["ann_vola"]] = ann_vol
                performance.loc[col, self.const_cols["sharpe_ratio"]] = sharpe
                performance.loc[col, self.const_cols["bench_corr"]] = corr

            ann_bret, ann_bvol, bsharpe, bcorr = \
            PortfolioGenerator(bench_rets.squeeze()).get_portfolio_performance(bench_rets.squeeze())

            performance.loc[self.const_cols["benchmark"], self.const_cols["ann_mean_ret"]] = ann_bret * 100
            performance.loc[self.const_cols["benchmark"], self.const_cols["ann_vola"]] = ann_bvol
            performance.loc[self.const_cols["benchmark"], self.const_cols["sharpe_ratio"]] = bsharpe
            performance.loc[self.const_cols["benchmark"], self.const_cols["bench_corr"]] = bcorr

            performance = performance.round(2)
            performance.index.name = self.const_cols["port_types"]
            performance_table = performance.reset_index().to_dict("records")

            fig = go.Figure()
            for col in hist_df.columns:
                fig.add_trace(go.Scatter(
                    x=cum_hist.index, y=cum_hist[col],
                    mode="lines", name=col))

                fig.add_trace(go.Scatter(
                    x=cum_fut.index, y=cum_fut[col],
                    mode="lines", name=col, line=dict(dash='dash')))

            fig.add_trace(go.Scatter(
                x=bench_cum.index, y=bench_cum,
                mode="lines", name=f"Benchmark {bench_cum.name}",
                line=dict(width=1)
            ))

            fig.update_layout(
                title=f"Portfolio performance {start.date()} → {end.date()}",
                xaxis_title="Date",
                yaxis_title="Cumulative return",
                template="plotly"
            )

            longpos_df = pd.DataFrame()

            for ptype, weights_dict in weights.items():
                if ptype != longpos_port_type:
                    continue

                act_w, act_pos = PortfolioGenerator(self.stock_rets).get_actual_invest(
                    weights_dict, self.actual_quotes
                )

                df = pd.DataFrame(index=constituents)
                df.index.name = self.const_cols["asset"]

                for tick in constituents:
                    df.loc[tick, self.const_cols["opt_weight"]] = weights_dict.get(tick, 0)
                    df.loc[tick, self.const_cols["act_weight"]] = act_w.get(tick, 0)
                    df.loc[tick, self.const_cols["long_pos"]] = act_pos.get(tick, [0])[0]
                    df.loc[tick, self.const_cols["amount"]] = act_pos.get(tick, [0, 0])[1]

                longpos_df = df.round(2).reset_index()

            longpos_data = longpos_df.to_dict("records")

            return fig, performance_table, longpos_data

    def run(self, debug=True):

        def run_dash():
            self.app.run_server(debug=debug, use_reloader=False)

        dash_thread = threading.Thread(target=run_dash)
        dash_thread.start()
        webbrowser.open_new("http://127.0.0.1:8050/")

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
    
    def _return_cleaning(self, df, col_filter):
        df_clean = df.iloc[1:]
        df_clean = df_clean.fillna(0)
        try:
            df_clean = df_clean[col_filter]
        except:
            pass
        return df_clean
    
    def _get_matching_keys(self, mapping_dict, base_list):
        matching_keys = [key for key, value in mapping_dict.items() \
                         if value in base_list]
        return matching_keys
    
    def _normalize_custom_weights(self, custom_weights, constituents):
        """
        Takes the current custom weights, keeps only the ones belonging
        to the selected constituents, and rescales them so total = 1.

        Keeps user input when constituents shrink or expand.
        """

        if not custom_weights:
            return {c: 0 for c in constituents}

        filtered = {k: custom_weights.get(k, 0) for k in constituents}
        s = sum(filtered.values())

        if s == 0:
            return filtered

        return {k: v / s for k, v in filtered.items()}
    
    def _validate_custom_weights(self, custom_weights):
        """
        Returns: (is_valid, deviation_float)
        """

        s = sum(custom_weights.values())
        deviation = round(1 - s, 3)
        if s == 0:
            valid = False
        else:
            valid = abs(deviation) < 0.0001

        return valid, deviation