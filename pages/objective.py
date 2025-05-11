import dash
from dash import html

dash.register_page(__name__, path="/objective")

layout = html.Div([
    html.P("Semiconductor materials are essential to modern technology — powering everything from processors and sensors to solar panels. "
           "However, discovering new materials with optimal properties is a time-consuming process. Traditional simulation techniques like "
           "Density Functional Theory (DFT), while accurate, can take hours or even days to evaluate a single compound. This limits the pace of innovation."),

    html.P("The goal of this project is to accelerate semiconductor material discovery by using machine learning (ML) models to predict "
           "one of the most critical properties of these materials: the band gap. We aim to achieve prediction performance comparable to DFT, "
           "but at a fraction of the computational cost and time — enabling rapid, large-scale screening of materials."),

    html.H4("Data Sources", style={"marginTop": "30px", "color": "#054b9c"}),
    html.P("We trained our models using two main sources of data:"),
    html.Ul([
        html.Li([
            html.B("Materials Project Database: "),
            "Provided structural and atomic features for over 170,000 materials, including space group number, density, crystal system, volume, and number of atomic sites. "
            "We also extracted physical descriptors from chemical formulas using ",
            html.Code("pymatgen"),
            ", such as weighted atomic number, electronegativity, and atomic radius."
        ]),
        html.Li([
            html.B("ORCA (DFT Calculations): "),
            "Used to calculate band gap values with high precision. These DFT-based results served as ground truth labels for supervised learning. "
            "In some cases, additional outputs like electron density and molecular volume were also integrated."
        ])
    ]),

    html.P("After cleaning and feature engineering, the final dataset included over 170,000 labeled materials with nine engineered features — ready for model training."),

    html.H4("Approach", style={"marginTop": "30px", "color": "#054b9c"}),
    html.P("We trained a range of models — from linear regression and ensemble methods like Random Forest and XGBoost, to deep learning models including "
           "fully connected neural networks (FCNNs) and graph neural networks (GNNs). GNNs were particularly valuable due to their ability to represent materials "
           "as atom-bond graphs, capturing structural relationships missed by tabular models."),

    html.H4("What This Enables", style={"marginTop": "30px", "color": "#054b9c"}),
    html.Ul([
        html.Li("⚡ Fast, accurate band gap prediction for unseen materials"),
        html.Li("📊 Reduced dependence on expensive DFT simulations"),
        html.Li("🚀 Scalable, high-throughput screening across thousands of candidates"),
        html.Li("🧪 More efficient, cost-effective material development workflows")
    ])
], style={"padding": "2rem"})
