import dash
from dash import html, dcc
import plotly.express as px
import pandas as pd

dash.register_page(__name__, path="/findings")

layout = html.Div([
    html.H2("Major Findings")
])