import dash
from dash import html
import pandas as pd

dash.register_page(__name__, path="/methods")


def styled_table(header, rows):
    return html.Table([
        html.Thead(html.Tr([
            html.Th(col, style={"padding": "10px", "backgroundColor": "#f2f2f2", "border": "1px solid #ccc"}) for col in header
        ])),
        html.Tbody([
            html.Tr([
                html.Td(cell, style={"padding": "10px", "border": "1px solid #ddd"}) for cell in row
            ], style={"backgroundColor": "#fff" if i % 2 == 0 else "#f9f9f9"})
            for i, row in enumerate(rows)
        ])
    ], style={
        "width": "100%",
        "borderCollapse": "collapse",
        "marginTop": "10px",
        "marginBottom": "30px",
        "boxShadow": "0 2px 8px rgba(0, 0, 0, 0.05)"
    })

layout = html.Div([
    html.P("When we started this project, we wanted to understand how AI could be used to predict the band gap of materials — "
           "a critical property in semiconductors. We knew that simulations like ORCA, which use Density Functional Theory (DFT), "
           "are accurate but slow. So, we decided to train machine learning models to do similar predictions faster."),

    html.P("Below is a breakdown of our process, decisions, and the tools we used along the way."),

    html.H3("Data Cleaning and processing:", style={"color": "#074185"}),

    html.P([ "We began with a dataset of over ",html.B("178,000"),
        " entries from Materials Project. After removing rows with missing band gap values and duplicate entries, we ended up with around ",
        html.B("170,000")," usable data points."]),
    html.P("We also:"),
    html.Ul([
        html.Li("Encoded 'Crystal System' into numerical format"),
        html.Li("Removed outliers and inconsistencies"),
        html.Li("Standardized feature formatting and scales")
    ]),

    html.P("The raw 'Formula' field wasn’t useful by itself, so we extracted physical and chemical features from it, including weighted atomic number, electronegativity, and atomic radius — to give our models more physical context."),

    html.P(["Simply feeding raw strings like ",html.B("Fe2O3")," or ",
        html.B("SiO2")," into a model is not helpful — models cannot interpret chemical formulas as-is. Instead, we used domain knowledge and ",
        html.Code("pymatgen"), " to extract weighted atomic properties from each formula. Specifically, we calculated:"
    ]),

    html.Ul([
        html.Li([html.B("Weighted atomic number"),": captures the average number of protons across all atoms in the formula"]),
        html.Li([html.B("Weighted electronegativity"),": reflects the compound’s tendency to attract electrons, affecting electronic behavior"]),
        html.Li([html.B("Weighted atomic radius"),": represents average atomic size, helping models infer structural or bonding characteristics"])
    ]),

    html.P([
        "Example: How weighted atomic number was calculated: ",
        html.Code("weighted_atomic_number = sum(element.atomic_number * fraction for each element in compound)")
    ]),

    html.P("These properties were computed by averaging the contributions of each element in a compound, weighted by its proportion in the formula. "
           "The resulting numerical features allow the model to capture the material’s chemistry in a structured and learnable way. "
           "This transformation was critical to making the dataset suitable for predictive modeling."),

    html.P("We also incorporated additional features such as density, volume, number of atomic sites, crystal system, and space group number. "
           "These gave the models more context about both the structure and composition of each material."),

    html.H5("Preprocessing Pipeline Summary"),
    html.Ol([
        html.Li("Removed missing values and duplicates"),
        html.Li("Parsed chemical formulas using Pymatgen"),
        html.Li("Extracted weighted atomic properties (e.g., atomic number, radius, electronegativity)"),
        html.Li("Encoded categorical features like Crystal System"),
        html.Li("Standardized all numeric features for model input")
    ]),

    html.P("Before cleaning:"),
    html.Img(src="/assets/before-clean.png", style={"display": "block", "margin": "0 auto", "maxWidth": "1000px", "marginBottom": "40px"}),
    html.P("Afrer cleaning:"),
    html.Img(src="/assets/after-clean.png", style={"display": "block", "margin": "0 auto", "maxWidth": "1000px", "marginBottom": "40px"}),

    html.H3("Models Training & Comparison", style={"color": "#074185"}),
    html.P("We tested a variety of models to figure out which one could best predict band gaps in semiconductor materials. Our process included everything from simple models to more advanced deep learning methods."),

    html.H5("Traditional & Ensemble Models"),
    html.P("We started with linear regression and ridge regression as basic baselines. They were quick to train and easy to understand, but they didn’t do a great job with the complex patterns in our data."),
    html.P("Next, we moved on to Random Forest and XGBoost. These performed way better because they’re good at handling nonlinear relationships and capturing how different features interact."),
    html.P("To make them even better, we used grid search to tune key settings like the number of trees (n_estimators) and how deep they can go (max_depth). We plotted these results as heatmaps to see which combinations gave us the lowest errors."),

    html.H5("Deep Learning Models"),
    html.P("We also built fully connected neural networks (FCNNs) to handle more complex patterns. But our best results came from graph neural networks (GNNs), which treat atoms and bonds as nodes and edges in a graph."),
    html.P("We implemented our GNN using PyTorch Geometric. While it required more effort to set up, the accuracy gains made it worth it."),
    html.P([
        "FCNN Architecture Summary: ",
        html.Code("Input → Dense(128) → ReLU → Dropout → Dense(64) → ReLU → Output")
    ]),

    html.H5("Model Comparison Summary"),
    styled_table(
        header=["Model Type", "Examples", "Why We Used It"],
        rows=[
            ["Traditional", "Linear Regression, Ridge", "Quick and easy starting point"],
            ["Ensemble", "Random Forest, XGBoost", "Great for handling complex feature combos"],
            ["Deep Learning", "FCNN, GNN", "Best at learning deep patterns and structure"]
        ]
    ),

    html.H5("Comparing with Traditional Simulations"),
    html.P("To see how our models stacked up against real-world tools, we compared them to ORCA, a quantum chemistry tool that uses DFT (density functional theory)."),
    styled_table(
        header=["Method", "Pros", "Cons"],
        rows=[
            ["ORCA (DFT)", "Super accurate", "Very slow — takes hours per material"],
            ["Random Forest", "Fast and scalable", "Not quite as accurate as ORCA"],
            ["GNN", "High accuracy and structure-aware", "More complex to implement"]
        ]
    ),

    html.P("In terms of speed, Random Forest was able to predict thousands of band gaps in the time it took ORCA to calculate just a few. That shows how useful ML models can be for speeding up materials research."),

    html.H5("Evaluation Metrics"),
    html.P("We used three main metrics to evaluate our models:"),
    html.Ul([
        html.Li([html.B("MAE (Mean Absolute Error:"), " Average prediction error"]),
        html.Li([html.B("RMSE (Root Mean Squared Error:"), " Penalizes big mistakes more"]),
        html.Li([html.B("R² Score:"), " Tells us how well the model explains the variation in band gap values"])
    ]),

    html.P([
        "To ensure reliable and generalizable results, we used an ",
        html.Span("80/20 train-test split", style={"color": "red"}),
        " alongside ",
        html.Span("5-fold cross-validation", style={"color": "red"}),
        ". This combination helped validate our models consistently while minimizing the risk of overfitting."
    ]),

    html.H5("Limitations & Assumptions"),
    html.Ul([
        html.Li("Tabular models miss spatial relationships between atoms"),
        html.Li("GNNs can learn from molecular structure directly"),
        html.Li("Results depend heavily on quality of extracted features")
    ], style={"marginBottom": "40px"}),

    html.H3("Visualization Strategy", style={"color": "#074185"}),
    html.P("To help make sense of everything, we made a bunch of visualizations:"),
    html.Ul([
        html.Li("Bar chart showing MSEs for Random Forest, FCNN, and GNN"),
        html.Li("Comparison plots for SVR, K-Neighbors, and Gradient Boosting"),
        html.Li("Heatmap of Random Forest hyperparameter tuning"),
        html.Li("Scatter plot of predicted vs. actual band gaps"),
        html.Li("Speed comparison chart of Random Forest vs. ORCA")
    ], style={"marginBottom": "40px"}),

    html.H3("Tools and Libraries", style={"color": "#074185"}),
    html.Div([
        html.Ul([
            html.Li("scikit-learn – for traditional ML models"),
            html.Li("XGBoost – for fast boosting trees"),
            html.Li("PyTorch Geometric – to build our GNN"),
            html.Li("ORCA – for DFT simulations"),
            html.Li("Pandas, Matplotlib, Seaborn, Plotly – for data and visuals"),
            html.Li("GridSearchCV – for hyperparameter tuning")
        ])
    ], style={"marginBottom": "40px"}),

    html.Div([
        html.P(
            "Our project showed that machine learning models — especially GNNs — can be powerful tools for predicting semiconductor properties. "
            "They're not only much faster than traditional DFT simulations, but they can also be pretty accurate. That makes them a great option for speeding up material discovery in the future."
        )
    ], style={"marginBottom": "40px"}),

    html.H3("Exploratory Data Analysis (EDA)", style={"color": "#074185"}),
    html.Div([
        html.Div([
            html.H5("Band Gap Distribution"),
            html.P("This histogram shows how most materials in our dataset have very low band gaps, indicating a predominance of metallic or narrow-bandgap materials."),
            html.P([
                "The box plot shows that most materials have band gaps below ",
                html.B("2.5 eV"), ", with a median around ", html.B("0.9 eV"), ". ",
                "There's a long tail of outliers above ", html.B("4.8 eV"), ", going all the way up to ", html.B("17.5 eV"), ". ",
                "This means the dataset has a lot of low-bandgap materials (like metals or semiconductors) and a few with very high gaps (likely insulators). ",
                "Because of the skewed distribution, it might help to normalize or transform the band gap values before feeding them into a machine learning model."
            ]),
            html.Img(src="/assets/bandgap.png", style={"display": "block", "margin": "0 auto", "maxWidth": "700px", "marginBottom": "40px"})
        ]),
        html.Div([
            html.H5("Feature Correlation Heatmap"),
            html.P(
                "This heatmap displays the pairwise Pearson correlation between numerical features in the dataset. "
                "Blue indicates negative correlation, red indicates positive correlation, and the numbers in each cell show "
                "the exact correlation coefficient. This helps identify which features might have predictive power or potential multicollinearity. "
            ),
            html.P([
                "We found some pretty strong correlations in the data — for example, ",
                html.Em("weighted atomic number"), " and ", html.Em("density"),
                " had a correlation of ", html.B("0.73"), ", which makes sense because heavier elements tend to be denser. ",
                "There was also a strong correlation between ", html.Em("number of sites"), " and ", html.Em("volume"),
                " (", html.B("0.80"), "), since more atomic sites usually mean a larger crystal structure. ",
                "Another interesting one was ", html.Em("weighted atomic number"), " and ", html.Em("weighted atomic radius"),
                " (", html.B("0.77"), "), showing that as atomic weight increases, the average atomic size tends to increase too."
            ]),
            html.P([
                "One additional insight from the heatmap is that ",
                html.Em("weighted electronegativity and weighted atomic radius"), " show a strong inverse relationship (", html.B("−0.86"),
                "), suggesting a trade-off between atomic size and electron-attracting behavior. ",
                "This level of correlation implies that including both features in certain models could introduce ",
                html.Em("feature redundancy"),
                ". In such cases, dimensionality reduction or feature selection may be helpful."
            ]),
            html.P([
                "Another takeaway is that no single feature is highly correlated with the band gap — the strongest correlation is around ",
                html.B("0.40"),
                ". This reinforces the idea that ",
                html.Em("band gap is influenced by a combination of structural, atomic, and electronic factors"),
                ", making it a suitable target for more expressive models like ensemble methods or graph neural networks."
            ]),
            html.P([
                "Interestingly, features like ", html.Em("Crystal System"), " and ", html.Em("Space Group Number"),
                " show weak correlations with band gap (", html.B("0.07"), " and ", html.B("−0.21"), "), indicating that ",
                html.Em("symmetry-based descriptors alone may not be strong predictors"),
                " unless paired with other structural details."
            ]),
            html.Img(src="/assets/corr-heatmap.png", style={"display": "block", "margin": "0 auto", "maxWidth": "700px", "marginBottom": "40px"})
        ]),
        html.Div([
            html.H5("Feature Correlation with Band Gap"),
            html.P([
                "This bar chart illustrates how individual features correlate with the ",
                html.Em("band gap"), " values across the dataset. Among all features, ",
                html.Em("weighted electronegativity"), " shows the strongest positive correlation (≈ ",
                html.B("0.40"), "), suggesting that materials with more electronegative elements are more likely to have higher band gaps — likely due to stronger electron localization and interactions."
            ]),
            html.P([
                html.Em("Sites"), " and ", html.Em("volume"),
                " also show moderate positive correlations (≈ ", html.B("0.31"), " and ", html.B("0.22"), "), which may reflect the fact that larger, more complex crystal structures can support wider band gaps."
            ]),
            html.P([
                "Conversely, ", html.Em("weighted atomic radius"), " (≈ ", html.B("−0.38"), "), ",
                html.Em("density"), " (≈ ", html.B("−0.31"), "), and ",
                html.Em("weighted atomic number"), " (≈ ", html.B("−0.29"), ") are negatively correlated with band gap, indicating that heavier, more tightly packed atoms tend to lead to smaller band gaps — a trend often observed in conductive or metallic materials."
            ]),
            html.P([
                html.Em("Crystal system"), " and ", html.Em("space group number"),
                " exhibit weak correlations, suggesting that symmetry alone is not a strong predictor of band gap and may need to be combined with other descriptors to be useful."
            ]),
            html.Img(src="/assets/features-corr-w-bandgap.png", style={"display": "block", "margin": "0 auto", "maxWidth": "700px" , "marginBottom": "40px"})
        ]),
        html.Div([
            html.H5("True vs. Predicted Band Gaps (Random Forest)"),
            html.P([
                "This scatter plot compares the ", html.Em("true"), " band gap values against the ",
                html.Em("predicted"), " values produced by the Random Forest model. Each point represents a single material, and the color gradient indicates the ",
                html.Em("residual error"), " — calculated as ", html.Em("(True - Predicted)"), "."
            ]),
            html.P([
                "Points near the red dashed line (ideal fit) represent predictions that are close to perfect. ",
                html.Span("Red shades indicate underpredictions", style={"color": "red"}),
                ", where the model underestimated the band gap. ",
                html.Span("Blue shades indicate overpredictions", style={"color": "blue"}),
                ", where the predicted value was higher than the true value."
            ]),
            html.P([
                "The model achieved an ", html.Em("R² score of "), html.B("0.56"),
                ", indicating that it captured over half the variance in the band gap values. "
                "However, the spread in residuals increases as the true band gap gets larger, suggesting some difficulty in predicting wide band gaps accurately."
            ]),
            html.P("This residual-based visualization helps pinpoint where the model struggles — particularly in the high-gap region — and can guide future improvements in feature engineering or model tuning."),
            html.Img(src="/assets/true vs predicted (RF).png", style={"display": "block", "margin": "0 auto", "maxWidth": "700px"})
        ])
    ]),
], style={"padding": "2rem"})