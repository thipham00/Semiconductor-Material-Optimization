import dash
from dash import html

dash.register_page(__name__, path="/methods")

layout = html.Div([
    html.H2("Analytical Methods"),
    html.P("We used machine learning models such as XGBoost and CNNs to analyze semiconductor properties."),
])

