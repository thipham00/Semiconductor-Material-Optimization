import dash
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc

app = Dash(
    __name__,
    use_pages=True,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.LUX]
)

server = app.server  # For deployment

# Header
header = html.Div([
    html.Div("Semiconductor Material Optimization", className="logo-title"),
    html.Div([
        dbc.Nav([
            dbc.NavLink("Home", href="/", id="main-link", className="nav-link", active="exact"),
            dbc.NavLink("Project Objective", href="/objective", id="objective-link", className="nav-link", active="exact"),
            dbc.NavLink("Analytical Methods", href="/methods", id="methods-link", active="exact"),
            dbc.NavLink("Major Findings", href="/findings", id="findings-link", active="exact"),
        ], className="navbar-row")
    ], className="navbar-wrapper")
], className="header-bar")

# Footer
footer = html.Footer([
    html.Div([
        html.P([
            "Project Github — ",
            html.A("Semiconductor Material Optimization",
                   href="https://github.com/thipham00/Semiconductor-Material-Optimization",
                   target="_blank")
        ]),
        html.P([
            "Thi Dao Nguyen Pham — ",
            html.A("thidaonguyen.pham@gmail.com, "),
            html.A("LinkedIn", href="https://www.linkedin.com/in/thi-dao-nguyen-pham/", target="_blank")
        ]),
        html.P([
            "Jewon Yeon — ",
            html.A("yeon971105@gmail.com, "),
            html.A("LinkedIn", href="https://www.linkedin.com/in/jewon-yeon-ai-scientist/", target="_blank")
        ])
    ])
], className="footer-bar")

# App layout
app.layout = html.Div([
    dcc.Location(id="url"),
    html.Div([
        header,
        html.Div(dash.page_container, className="content-col with-bg"),
        footer
    ], style={"minHeight": "100vh", "display": "flex", "flexDirection": "column"})
])

# Run app
if __name__ == '__main__':
    app.run(debug=True)
