import dash
from dash import html

dash.register_page(__name__, path="/", name="Home")

layout = html.Div([
    html.Div([
        html.H1("Semiconductor Material Optimization", style={
            "color": "white",
            "fontSize": "62px",
            "fontWeight": "800",
            "textAlign": "center",
            "marginBottom": "20px",
            "textShadow": "0 2px 4px rgba(0,0,0,0.5)"
        }),

        html.P("Accelerating material discovery through machine learning and simulation",
               style={
                   "color": "#dce5ec",
                   "fontSize": "20px",
                   "textAlign": "center",
                   "marginBottom": "50px",
                   "textShadow": "0 1px 2px rgba(0,0,0,0.4)"
               }),

        html.Div([
            html.Div([
                html.H4("🔬 Project Objective", style={"color": "#00f0ff", "fontWeight": "700"}),
                html.P("Apply data science to understand and predict semiconductor material performance.")
            ], className="home-card"),

            html.Div([
                html.H4("📊 Analytical Methods", style={"color": "#00f0ff", "fontWeight": "700"}),
                html.P("Use Random Forest, XGBoost, FCNN, and GNN models for band gap prediction.")
            ], className="home-card"),

            html.Div([
                html.H4("⚡ Major Findings", style={"color": "#00f0ff", "fontWeight": "700"}),
                html.P("Ensemble models outperform deep learning on structured tabular data.")
            ], className="home-card")
        ], style={
            "display": "flex",
            "justifyContent": "space-around",
            "flexWrap": "wrap",
            "gap": "20px",
            "maxWidth": "1100px",
            "margin": "auto"
        })
    ], style={
        "backgroundColor": "rgba(0,0,0,0.6)",
        "padding": "60px 20px",
        "borderRadius": "12px",
        "maxWidth": "1200px",
        "margin": "60px auto"
    })
])


