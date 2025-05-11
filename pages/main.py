import plotly.graph_objects as go
import pandas as pd
import numpy as np
import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc

dash.register_page(__name__, path="/")

# Data
results_df = pd.DataFrame({
    "Model": [
        "Random Forest",
        "Gradient Boosting",
        "Linear Regression",
        "Support Vector Regression",
        "K-Neighbors"
    ],
    "MSE": [0.9746, 1.3911, 1.6805, 1.4912, 1.2497],
    "R2 Score": [0.5606, 0.3728, 0.2423, 0.3277, 0.4366]
})

def create_figure(metric="MSE"):
    max_val = results_df[metric].max()
    y_range = [0, round(max_val + 0.1, 1)]
    colors = ['steelblue', 'indianred', 'seagreen', 'darkorange', 'mediumslateblue']
    fig = go.Figure(data=[
        go.Bar(
            x=results_df["Model"],
            y=results_df[metric],
            marker_color=colors,
            text=[f"{v:.2f}" for v in results_df[metric]],
            textposition='auto'
        )
    ])
    fig.update_layout(
        title=f"Model Comparison - {metric}",
        xaxis_title="Model",
        yaxis_title=metric,
        plot_bgcolor="#ecf0f1",
        paper_bgcolor="#ecf0f1",
        margin=dict(t=50, b=50, l=50, r=50),
        xaxis=dict(showgrid=True, gridcolor='#bcbcbc'),
        yaxis=dict(showgrid=True, gridcolor='#bcbcbc', range=y_range),
        showlegend=False
    )
    return fig

@callback(
    Output("comparison-chart", "figure", allow_duplicate=True),
    Input("metric-dropdown", "active_tab"),
    prevent_initial_call=True
)
def update_chart(selected_metric):
    return create_figure(selected_metric)

# Chart 2
model_names = ['Ensemble', 'FCNN', 'GNN']
test_mse_values = [0.8349, 1.6665, 1.88]
fig = go.Figure(data=[
    go.Bar(
        x=model_names,
        y=test_mse_values,
        text=[f'{v:.3f}' for v in test_mse_values],
        textposition='auto',
        marker_color=['steelblue', 'indianred', 'seagreen']
    )
])
fig.update_layout(
    title="Comparison of Test MSE for Three Models",
    xaxis=dict(title="Model", showgrid=True, gridcolor="#bcbcbc"),
    yaxis=dict(title="Test MSE", showgrid=True, gridcolor="#bcbcbc"),
    plot_bgcolor="#ecf0f1",
    paper_bgcolor="#ecf0f1"
)

# Chart 3
num_atoms = np.arange(1, 1001, 10)
orca_time = 0.00001 * (num_atoms ** 3)
rf_time = 0.001 * num_atoms
time_fig = go.Figure()
time_fig.add_trace(go.Scatter(x=num_atoms, y=orca_time, mode='lines', name='ORCA (DFT)', line=dict(color='red')))
time_fig.add_trace(go.Scatter(x=num_atoms, y=rf_time, mode='lines', name='Ensemble Model (ML)', line=dict(color='blue')))
time_fig.update_layout(
    title="Number of Atoms vs Time Comparison (ORCA vs Ensemble Model)",
    xaxis=dict(title="Number of Atoms", range=[0, 1000], showgrid=True, gridcolor='#bcbcbc', zeroline=False),
    yaxis=dict(title="Time (seconds)", type="log", range=[-2, 5], showgrid=True, gridcolor='#bcbcbc', zeroline=False),
    plot_bgcolor="#ecf0f1",
    paper_bgcolor="#ecf0f1",
    legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)", bordercolor="#bcbcbc", borderwidth=1),
    margin=dict(t=60, b=50, l=50, r=50)
)

layout = html.Div([
    html.Div([
        html.Div([
            html.Div([
                html.Img(src="/assets/orca.png", style={"height": "80px", "marginBottom": "15px"}),
                html.P("Used for high-accuracy quantum chemistry calculations, ORCA helped us generate band gap values through DFT simulations.",
                       style={"color": "#444"}),
                html.A("Visit ORCA Website →", href="https://orcaforum.kofo.mpg.de/", target="_blank",
                       style={"marginTop": "15px", "display": "inline-block", "color": "#1E3A8A", "textDecoration": "none", "fontWeight": "500"})
            ], style={
                "padding": "15px", "textAlign": "center", "backgroundColor": "#ffffff",
                "borderRadius": "12px", "boxShadow": "0 4px 12px rgba(0, 0, 0, 0.08)", "width": "100%"
            }),
            html.Div([
                html.Img(src="/assets/mp.png", style={"height": "80px", "marginBottom": "15px"}),
                html.P("Our core dataset came from the Materials Project API, including structural and atomic information on thousands of materials.",
                       style={"color": "#444"}),
                html.A("Visit Materials Project Website→", href="https://materialsproject.org/", target="_blank",
                       style={"marginTop": "15px", "display": "inline-block", "color": "#1E3A8A", "textDecoration": "none", "fontWeight": "500"})
            ], style={
                "padding": "15px", "textAlign": "center", "backgroundColor": "#ffffff",
                "borderRadius": "12px", "boxShadow": "0 4px 12px rgba(0, 0, 0, 0.08)", "width": "100%"
            })
        ], style={
            "display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(280px, 1fr))",
            "gap": "30px", "padding": "20px", "backgroundColor": "#f0f4f8",
            "borderRadius": "12px",  "marginTop": "0px", "marginBottom": "30px"
        })
    ]),
    html.P("Welcome to our project dashboard on Semiconductor Material Optimization. In this project, we explore how artificial intelligence, "
           "particularly machine learning and deep learning techniques, can be used to predict one of the most essential properties of semiconductor materials — "
           "the band gap. The goal is to accelerate the material discovery process by replacing slow, traditional simulation techniques like Density Functional Theory (DFT) "
           "with fast, scalable, and accurate predictive models."),
    html.P("Semiconductors form the foundation of modern electronics, from processors to solar cells. Discovering and optimizing new materials with specific electrical "
           "properties is a crucial part of innovation in fields like clean energy, microelectronics, and nanotechnology. However, existing computational methods "
           "like DFT can take several hours or even days to simulate the properties of a single compound. Our project proposes a machine-learning-driven alternative — "
           "predicting material band gaps using structured atomic and physical data, with accuracy comparable to DFT but orders of magnitude faster.",),
    html.H3("Project Highlights", style={"marginTop": "40px", "color": "#054b9c"}),
    html.Ul([
        html.Li("✅ Used a dataset of over 170,000 materials sourced from public repositories and simulations (Materials Project, OQMD, ORCA)"),
        html.Li("✅ Engineered features from atomic formulas to feed into predictive models"),
        html.Li("✅ Compared multiple models: Linear Regression, Random Forest, XGBoost, FCNN, and Graph Neural Networks (GNNs)"),
        html.Li("✅ Achieved the best performance using GNNs, capable of modeling atomic structure directly"),
        html.Li("✅ Validated model accuracy with metrics like MAE, RMSE, and R²"),
        html.Li("✅ Benchmarked speed and scalability vs. traditional simulation software like ORCA"),
        html.Li("✅ Developed an interactive tool where users can input material properties and get predicted band gaps instantly")
    ]),
    html.H3("Why This Matters", style={"marginTop": "40px", "color": "#054b9c"}),
    html.P("Machine learning has the potential to reshape how we approach material science. Rather than relying solely on expensive, time-intensive simulations, "
           "we can train models to learn from past calculations and generalize to new compounds — significantly speeding up the screening process. This makes it possible "
           "to evaluate thousands of potential materials in minutes instead of weeks."),

    html.P("By focusing on the band gap — a key factor in determining whether a material is a conductor, semiconductor, or insulator — we can guide researchers toward "
           "materials that fit the criteria for high-performance electronic applications. Our work demonstrates that predictive modeling not only saves time but can also "
           "improve understanding of what factors most influence material behavior."),

    html.H3("Visual Overview", style={"marginTop": "40px", "color": "#054b9c"}),

    html.H5("Regression Model Comparison (MSE & R² Score)", style={"marginTop": "20px"}),
    html.Div([
        dbc.Tabs(
            [
                dbc.Tab(label="Mean Squared Error (MSE)", tab_id="MSE", tab_style={"textAlign": "center"}),
                dbc.Tab(label="R² Score", tab_id="R2 Score", tab_style={"textAlign": "center"})
            ],
            id="metric-dropdown",
            active_tab="MSE",
            className="custom-tabs"
        )
    ], className="tab-wrapper"),
    dcc.Graph(id="comparison-chart", style={"marginBottom": "40px"}),

    html.H5("Comparison of Test MSE for Ensemble, FCNN, and GNN"),
    dcc.Graph(figure=fig, style={"marginBottom": "40px"}),

    html.H5("ORCA vs Stacking Ensemble — Computational Efficiency and Accuracy"),
    dcc.Graph(figure=time_fig),
], style={"padding": "2rem"})
