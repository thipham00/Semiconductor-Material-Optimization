import dash
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc

# Initialize Dash app with Bootstrap styles
app = Dash(__name__, external_stylesheets=[dbc.themes.LUX], use_pages=True)

# Expose the Flask server for Gunicorn
server = app.server

# Sidebar for navigation
sidebar = dbc.Col(
    dbc.Nav(
        [
            dbc.NavLink("Home", href="/", id="main-link", className="nav-link"),
            dbc.NavLink("Project Objective", href="/objective", id="objective-link", className="nav-link"),
            dbc.NavLink("Analytical Methods", href="/methods", id="methods-link", className="nav-link"),
            dbc.NavLink("Major Findings", href="/findings", id="findings-link", className="nav-link"),
        ],
        vertical=True,
        pills=True,
        className="sidebar",
    ),
    width=2,  # Sidebar takes 2 out of 12 columns
    className="sidebar-col"
)

# Main content area
content = dbc.Col(
    html.Div(
        [
            html.H1("Semiconductor Material Optimization", className="page-title", style={
                "backgroundColor": "#E0F2FE",
                "color": "#1E3A8A",
                "padding": "20px",
                "borderRadius": "8px",
                "boxShadow": "0 4px 8px rgba(0, 0, 0, 0.1)",
                "textAlign": "center",
                "marginBottom": "20px",
            }),
            dash.page_container,  # Loads the active page
        ]
    ),
    width=9,
    className="content-col"
)

# App layout
app.layout = html.Div([
    dbc.Row([sidebar, content])  # Arrange sidebar and main content
])


# Run the app
if __name__ == '__main__':
    app.run(debug=True)

