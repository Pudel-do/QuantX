from dash import html, dcc, dash_table
from dashboard.layout_data import (
    load_active_assets,
    load_slider_values
)

def create_layout():
    return html.Div(
        children=[
            html.H1("📊 Financial Dashboard", style={"textAlign": "center"}),

            html.P(" ", style={'margin': '60px 0'}),
            html.H2("Market Analysis"),
            build_market_section(),
            html.Hr(),

            html.P(" ", style={'margin': '60px 0'}),
            html.H2("Portfolio Analysis"),
            build_portfolio_section(),
            html.Hr()
        ]
    )

def build_market_section():
    cfg = load_slider_values()
    
    return html.Div(
        [   
            dcc.RangeSlider(
                id="return_slider",
                min=cfg.get("min"),
                max=cfg.get("max"),
                value=cfg.get("value"),
                marks=cfg.get("marks"),
                step=cfg.get("step"),
                allowCross=False
            ),

            dcc.Graph(id="cum_returns"),

            html.Div(
                id="performance_annotation",
                style={
                    "fontWeight": "bold",
                    "marginBottom": "6px",
                    "fontSize": "14px"
                }
            ),

            html.P(" ", style={'margin': '40px 0'}),
            
            dash_table.DataTable(
                id="performance_table",
                page_size=10,
                style_cell={"textAlign": "left"},
                style_header={"fontWeight": "bold"},
            ),

            html.P(" ", style={'margin': '40px 0'}),

            dcc.Graph(id="corr_heatmap")
            
        ]
    )

def build_portfolio_section():
    pass
