import dash
from dash import Dash, html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc

app = Dash(__name__, use_pages=True,suppress_callback_exceptions=True,external_stylesheets=[dbc.themes.LUX])
server = app.server

# Navbar
navbar = html.Div([
    dbc.Nav([
        dbc.NavLink("Home", href="/", id="main-link", className="nav-link", active="exact"),
        dbc.NavLink("Project Objective", href="/objective", id="objective-link", className="nav-link", active="exact"),
        dbc.NavLink("Analytical Methods", href="/methods", id="methods-link", className="nav-link", active="exact"),
        dbc.NavLink("Major Findings", href="/findings", id="findings-link", className="nav-link", active="exact"),
    ], className="navbar-row")
], className="navbar-wrapper")

# Main layout
app.layout = html.Div([
    dcc.Location(id="url"),
    html.Div(id="layout-wrapper")
])

# Callback to control title and background visibility
@callback(Output("layout-wrapper", "children"), Input("url", "pathname"))
def display_page(pathname):
    is_home = pathname == "/"
    children = []

    if not is_home:
        # Floating logo title at top-right
        children.append(html.Div("Semiconductor Material Optimization", className="logo-title"))
        children.append(navbar)
        children.append(html.Hr(className="section-divider"))
    else:
        children.append(navbar)

    children.append(dash.page_container)
    return html.Div(children, className="content-col with-bg" if not is_home else "content-col no-bg")


# main
if __name__ == '__main__':
    app.run(debug=True)
