from dash.dependencies import Input, Output, State, ALL
from dashboard.visuals.annotations import build_dash_annotation

def register_market_callbacks(app, adapter, styler):

    @app.callback(
        Output("cum_returns", "figure"),
        Output("performance_table", "data"),
        Output("performance_annotation", "children"),
        Input("return_slider", "value")
    )

    def update_market_section(slider):
        slider = tuple(slider)

        fig_cum_rets = adapter.build_cum_returns(slider)
        performance_table = adapter.build_return_peformance(slider)

        start, end = adapter.slider_to_dates(slider)

        return (
            fig_cum_rets,
            performance_table,
            build_dash_annotation(start, end)
    )

    @app.callback(
        Output("performance_table", "style_data_conditional"),
        Input("performance_table", "derived_virtual_data")
    )
    def style_table(rows):
        return styler.build_styles(rows)

    