import dash
from dash import html

dash.register_page(__name__, path="/objective")

layout = html.Div([
    html.H2("Project Objective"),
    html.P("The main goal of this project is to optimize semiconductor materials for improved efficiency."),
])


