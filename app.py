import dash
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc

# Enable multi-page functionality
app = Dash(__name__, external_stylesheets=[dbc.themes.LUX], use_pages=True)

# Define Layout
app.layout = html.Div([
    # Sidebar for navigation
    dbc.Row([
        dbc.Col(
            dbc.Nav(
                [
                    dbc.NavLink("Home", href="/", id="home-link", className="nav-link"),
                    dbc.NavLink("EDA", href="/eda", id="eda-link", className="nav-link"),
                    dbc.NavLink("Proposal", href="/proposal", id="proposal-link", className="nav-link"),
                ],
                vertical=True,
                pills=True,
                className="sidebar"
            ),
            width=2,  # Sidebar takes 2 out of 12 cols
            className="sidebar-col"
        ),
        # Main Content Area
        dbc.Col([
            # Title with light background and dark text
            html.Div(
                html.H1("Semiconductor Material Optimization", className="page-title"),
                style={
                    "backgroundColor": "#E0F2FE",  # Light blue background
                    "color": "#1E3A8A",  # Dark blue text
                    "padding": "20px",
                    "borderRadius": "8px",
                    "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.1)",
                    "textAlign": "center",
                    "marginBottom": "20px",
                }
            ),

            # Page Content (dynamic based on selected page)
            dash.page_container,
        ], width=9, className="content-col"),
    ]),
])

if __name__ == '__main__':
    app.run(debug=True)

