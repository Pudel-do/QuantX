from dash import Dash
from dashboard.layout import create_layout

def create_app():
    app = Dash(
        __name__, 
        suppress_callback_exceptions=True
    )
    app.title = "Financial Dashboard"

    app.layout = create_layout()

    return app

app = create_app()
server = app.server

if __name__ == "__main__":
    app.run_server(
        debug=True,
        use_reloader=False
    )