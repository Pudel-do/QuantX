from dash.dependencies import Input, Output, State, ALL

def register_callbacks(app, adapter):

    @app.callback(
        Output("cum_returns", "figure"),
        Input("return_slider", "value")
    )

    def update_cum_returns(slider):
        return adapter.build_cum_returns(tuple(slider))