import dash
from dash import html

dash.register_page(__name__, path="/proposal", name="Proposal")

layout = html.Div([
    html.H2("Project Proposal", style={'textAlign': 'center'}),

    html.Iframe(
        src="https://docs.google.com/document/d/1P_dXrRVSyQmKgSOp0teCmQPSEX1UgSOtOf5Q9H0W4no/preview",  
        style={"width": "100%", "height": "90vh", "border": "none"}
    )
])
