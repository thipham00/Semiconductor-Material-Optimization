import dash
from dash import html

dash.register_page(__name__, path="/", name="Home")

layout = html.Div([
    html.H2("Welcome to Semiconductor Material Optimization")
])
