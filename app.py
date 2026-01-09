from dash import Dash
from dashboard.layout import create_layout
import webbrowser
import threading

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

def run(debug=True):

    def run_dash():
        app.run_server(
            debug=debug, 
            use_reloader=False
        )
    dash_thread = threading.Thread(target=run_dash)
    dash_thread.start()
    webbrowser.open_new("http://127.0.0.1:8050/")

if __name__ == "__main__":
    run()