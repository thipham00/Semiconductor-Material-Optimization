import dash
from dash import html
import pandas as pd

dash.register_page(__name__, path="/methods")

layout = html.Div([
    html.H2("Analytical Methods", style={"color": "#1E3A8A", "fontWeight": "700"}),

    html.P("""
        Our analysis uses a variety of machine learning techniques to predict semiconductor material properties, 
        particularly the band gap. One of the core models employed is the Random Forest algorithm, which takes in 
        structured, numeric tabular data such as density, volume, and atomic characteristics. This model is well-suited 
        for capturing non-linear relationships and interactions between features, making it a reliable choice for 
        scientific data prediction.
    """),

    html.P("""
        To optimize model performance, we apply Grid Search Hyperparameter Tuning. This method systematically evaluates 
        combinations of parameters — such as the number of estimators and the maximum depth of trees — to find the best 
        configuration. The goal is to minimize mean squared error (MSE) and maximize predictive accuracy for the Random 
        Forest model.
    """),

    html.P("""
        In addition to traditional machine learning, we experiment with deep learning architectures, including 
        Fully Connected Neural Networks (FCNN) and Graph Neural Networks (GNN). These models are designed to handle 
        both structured tabular data and graph-based representations of molecular structures. Our hypothesis is that 
        GNNs may better capture the dependencies between atoms and improve prediction accuracy.
    """),

    html.P("""
        We also compare the performance and efficiency of our machine learning approach with traditional quantum 
        chemistry simulations, specifically the ORCA software which uses Density Functional Theory (DFT). While ORCA 
        is highly accurate, it requires significant computation time for each molecule. In contrast, our Random Forest 
        model can rapidly predict band gaps across large datasets. This comparison helps illustrate the trade-off 
        between computational cost and prediction accuracy — and showcases how machine learning offers a scalable 
        alternative for accelerating materials research.
    """),

    html.H2("Visualization Strategy", style={"color": "#1E3A8A", "fontWeight": "700", "marginTop": "40px"}),

    html.P("""
        To support our analysis, we’ve designed several visualizations that make performance comparisons easier to 
        interpret. First, we compare the test MSE of Random Forest, FCNN, and GNN models using a bar chart to quickly 
        identify which model performs best.
    """),

    html.P("""
        A more detailed comparison involves plotting the MSE and R² scores of multiple regression models — including 
        Gradient Boosting, Linear Regression, Support Vector Regression (SVR), and K-Neighbors — giving us a broader 
        view of model strengths.
    """),

    html.P("""
        We also visualize hyperparameter tuning results with a heatmap showing the impact of parameter combinations 
        (like number of estimators and tree depth) on model performance. This helps us pinpoint the most effective 
        settings for Random Forest.
    """),

    html.P("""
        A scatter plot of true vs. predicted band gap values provides insight into the accuracy of the best model, 
        helping identify areas of under- or over-prediction.
    """),

    html.P("""
        Finally, we include a bar chart comparing the number of band gaps predicted by Random Forest in a given time 
        frame versus those computed by ORCA. This emphasizes the speed and scalability advantage of machine learning 
        for material screening and discovery.
    """)
], style={"padding": "20px 10px", "maxWidth": "1000px", "margin": "auto"})
