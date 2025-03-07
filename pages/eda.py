import dash
from dash import html

dash.register_page(__name__, path="/eda", name="EDA")

layout = html.Div([
    html.H2("Exploratory Data Analysis (EDA)")
])
