import plotly.express as px
import pandas as pd
import numpy as np
import dash
from dash import html, dcc, callback, Input, Output
import plotly.graph_objects as go

dash.register_page(__name__, path="/findings")

# result 1 chart
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

# Initial figure (MSE by default)
def create_figure(metric="MSE"):
    max_val = results_df[metric].max()
    y_range = [0, round(max_val + 0.1, 1)]
    custom_colors = ['steelblue', 'indianred', 'seagreen', 'darkorange', 'mediumslateblue']
    fig = go.Figure(data=[
        go.Bar(
            x=results_df["Model"],
            y=results_df[metric],
            marker_color=custom_colors,
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
        yaxis=dict(
            showgrid=True,
            gridcolor='#bcbcbc',
            tick0=0,
            dtick=0.1,
            range=y_range
        ),
        showlegend=False
    )
    return fig

# Callback
@dash.callback(
    Output("comparison-chart", "figure"),
    Input("metric-dropdown", "value")
)
def update_chart(selected_metric):
    return create_figure(selected_metric)

# result 2 chart
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
    xaxis=dict(
        title="Model",
        showgrid=True,
        gridcolor="#bcbcbc"
    ),
    yaxis=dict(
        title="Test MSE",
        range=[0, max(test_mse_values) * 1],
        showgrid=True,
        gridcolor="#bcbcbc"
    ),
    plot_bgcolor="#ecf0f1",
    paper_bgcolor="#ecf0f1"
)

# result 3 chart:
num_atoms = np.arange(1, 1001, 10)
orca_time = 0.00001 * (num_atoms ** 3)
rf_time = 0.001 * num_atoms
time_fig = go.Figure()

# ORCA (DFT) Line
time_fig.add_trace(go.Scatter(
    x=num_atoms,
    y=orca_time,
    mode='lines',
    name='ORCA (DFT)',
    line=dict(color='red', width=2)
))
# Ensemble (ML) Line
time_fig.add_trace(go.Scatter(
    x=num_atoms,
    y=rf_time,
    mode='lines',
    name='Ensemble Model (ML)',
    line=dict(color='blue', width=2)
))
#layout
time_fig.update_layout(
    title="Number of Atoms vs Time Comparison (ORCA vs Ensemble Model)",
    xaxis=dict(
        title="Number of Atoms",
        range=[0, 1000],
        showgrid=True,
        gridcolor='#bcbcbc',
        zeroline=False
    ),
    yaxis=dict(
        title="Time (seconds)",
        type="log",
        range=[-2, 5],  # log10 scale: 10^-2 to 10^5
        showgrid=True,
        gridcolor='#bcbcbc',
        minor=dict(showgrid=True),
        zeroline=False
    ),
    plot_bgcolor="#ecf0f1",
    paper_bgcolor="#ecf0f1",
    margin=dict(l=50, r=50, t=60, b=50),
    legend=dict(
        x=0.02,
        y=0.98,
        xanchor="left",
        yanchor="top",
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="#bcbcbc",
        borderwidth=1
    )
)

# main content layout
layout = html.Div([
    html.Div([
        html.H3("1: Stacking Ensemble Model Performance", style={"color": "#2c3e50"}),
        html.H4("Summary:", style={"marginTop": "10px", "color": "#34495e"}),
        html.P([
            "The Stacking Ensemble, combining ", html.B("Random Forest, XGBoost, Gradient Boosting, and K-Nearest Neighbors"),
            ", achieved a ",
            html.B("Mean Squared Error (MSE)", style={"color": "red"}), " of ",
            html.B("0.8349", style={"color": "red"}),
            ", Root Mean Squared Error (RMSE) of ",
            html.B("0.9137", style={"color": "red"}),
            ", and an R² score of ",
            html.B("0.5731", style={"color": "red"}),
            " on the test set. This performance surpassed individual models, including standalone Random Forest, "
            "indicating that ensemble learning can significantly enhance prediction accuracy in semiconductor band gap prediction."
        ]),

        html.H4("Implications:", style={"marginTop": "15px", "color": "#34495e"}),
        html.P("""
            This result supports the strategy of using ensemble methods to capture diverse patterns within
            the dataset. Future steps should focus on fine-tuning the meta-model or incorporating additional
            diverse models like Support Vector Regression or LightGBM. Achieving R² scores beyond 0.6
            would make machine learning a highly reliable alternative to traditional simulations in
            semiconductor R&D, potentially accelerating the discovery of next-generation materials.
        """),

        html.Div([
            html.Label("Select metric to compare:", style={"marginRight": "10px"}),
            dcc.Dropdown(
                id="metric-dropdown",
                options=[
                    {"label": "Mean Squared Error (MSE)", "value": "MSE"},
                    {"label": "R² Score", "value": "R2 Score"},
                ],
                value="MSE",
                clearable=False,
                style={"width": "300px"}
            )
        ], style={"marginBottom": "20px"}),

        dcc.Graph(id="comparison-chart")
    ], style={"marginBottom": "40px"}),

    html.Div([
        html.H3("2: FCNN and GNN Performance Compared to Ensemble Methods", style={"color": "#2c3e50"}),
        html.H4("Summary:", style={"marginTop": "10px", "color": "#34495e"}),
        html.P([
            "Deep learning models — the Fully Connected Neural Network (FCNN) and the Graph Neural "
            "Network (GNN) — performed worse than traditional ensemble methods. The FCNN achieved a "
            "test MSE of ",
            html.B("1.667", style={"color": "red"}),
            ", and the GNN achieved a test MSE of ",
            html.B("1.880", style={"color": "red"}),
            ", both significantly higher than the Stacking Ensemble’s MSE of ",
            html.B("0.8349", style={"color": "red"}),
            ". This result highlights that for this tabular material "
            "property dataset, ensemble models outperformed deep learning models."
        ]),

        html.H4("Implications:", style={"marginTop": "15px", "color": "#34495e"}),
        html.P("""
            This finding refutes the initial hypothesis that GNNs, by modeling structural dependencies,
            would outperform classical models. It shows that, at least for datasets structured primarily as
            tabular features, traditional ensemble techniques are more effective. Future directions include
            experimenting with hybrid architectures that fuse tabular feature learning with graph-based
            learning, or developing custom featurization methods to better exploit deep learning
            advantages.
        """),
        dcc.Graph(figure=fig, style={"marginBottom": "40px"})
    ], style={"marginBottom": "40px"}),

    html.Div([
        html.H3("3: ORCA vs Stacking Ensemble — Computational Efficiency and Accuracy", style={"color": "#2c3e50"}),
        html.H4("Summary:", style={"marginTop": "10px", "color": "#34495e"}),
        html.P([
            "Compared to ORCA's DFT-based calculations, which take minutes per molecule, the trained "
            "Stacking Ensemble could predict thousands of band gaps nearly instantaneously after training. "
            "Although ORCA still provides slightly higher precision for individual molecules, the ensemble "
            "model reached an R² of ",
            html.B("0.5731", style={"color": "red"}),
            ", demonstrating strong predictive power. With additional feature "
            "engineering and dataset expansion, future models are expected to push the R² closer to ",
            html.B("0.65–0.75", style={"color": "red"}),
            ", narrowing the gap between machine learning and full quantum chemistry "
            "simulations."
        ]),

        html.H4("Implications:", style={"marginTop": "15px", "color": "#34495e"}),
        html.P("""
            This result underscores the massive computational advantage of machine learning approaches
            for material discovery. Replacing or complementing DFT calculations with ensemble predictions
            could lead to huge cost and time savings in semiconductor research. Broadly, this democratizes
            access to material innovation by enabling even small research labs and startups to rapidly
            screen material candidates without requiring supercomputing resources.
        """),
        dcc.Graph(figure=time_fig)
    ])
], style={"padding": "20px 10px", "maxWidth": "1100px", "margin": "auto"})
