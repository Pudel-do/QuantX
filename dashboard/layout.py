from dash import html, dcc, dash_table
from dashboard.layout_data import (
    load_active_assets,
    load_return_slider_values
)

def create_layout():
    return html.Div(
        children=[
            html.H1("📊 Financial Dashboard", style={"textAlign": "center"}),

            html.P(" ", style={'margin': '60px 0'}),
            html.H2("Market Analysis"),
            build_market_section(),
            html.Hr()
        ]
    )

def build_market_section():
    marks, date_range = load_return_slider_values()
    
    return html.Div(
        [
            dcc.RangeSlider(
                id="return_slider",
                min=0,
                max=len(date_range) - 1,
                value=[0, len(date_range) - 1],
                marks=marks,
                step=1,
                allowCross=False
            ),
        ]
    )

def style_table(
    self,
    id,
    page_size=10,
    ):
    """
    Einheitlich gestylte Dash DataTable (nur statisches Styling).
    Dynamische Färbung erfolgt vollständig über Callbacks.
    """

    style_data_conditional = [
        # Zebra-Streifen
        {
            "if": {"row_index": "odd"},
            "backgroundColor": "#fafafa",
        },

        # Hover / Active
        {
            "if": {"state": "active"},
            "backgroundColor": "#e6f2ff",
            "border": "1px solid #3399ff",
        },
    ]

    return dash_table.DataTable(
        id=id,
        page_size=page_size,

        style_header={
            "fontWeight": "bold",
            "backgroundColor": "#f2f4f8",
            "borderBottom": "2px solid #b0b0b0",
            "textAlign": "center",
            "fontSize": "14px",
        },

        style_cell={
            "padding": "8px",
            "fontSize": "13px",
            "fontFamily": "Segoe UI, Arial",
            "border": "1px solid #e1e1e1",
            "textAlign": "right",
            "whiteSpace": "normal",
            "height": "auto",
        },

        style_cell_conditional=[
            {
                "if": {"column_id": self.const_cols["asset"]},
                "textAlign": "left",
                "fontWeight": "bold",
            }
        ],

        style_data_conditional=style_data_conditional,
    )