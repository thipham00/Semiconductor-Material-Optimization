import dash
from dash import html

dash.register_page(__name__, path="/objective")

layout = html.Div([
    html.H2("Project Summary", style={"color": "#1E3A8A", "fontWeight": "700"}),

    html.P("""
        Semiconductor materials are at the heart of modern electronics, powering everything from smartphones to solar panels. 
        However, identifying and refining new materials is a slow, resource-intensive process. Traditional methods like 
        density functional theory (DFT) offer precision but demand substantial computational power.
    """),

    html.P("""
        This project seeks to revolutionize semiconductor material discovery by integrating advanced machine learning (ML) 
        and deep learning models. By leveraging tools like the ORCA quantum chemistry package to generate data on band gaps, 
        electron density, and molecular volume, we build predictive models that can evaluate material viability in seconds — 
        not hours or days.
    """),

    html.P("""
        Our goal is simple: reduce waste, improve efficiency, and accelerate innovation. Using ensemble methods and graph 
        neural networks (GNNs), we aim to predict semiconductor performance with high accuracy. This enables rapid screening 
        of new materials, helping manufacturers minimize trial-and-error and make smarter, faster decisions.
    """),

    html.H2("Broader Impacts", style={"color": "#1E3A8A", "fontWeight": "700", "marginTop": "40px"}),

    html.Ul([
        html.Li(html.B("🌱 Sustainability")),
        html.P("""
            By reducing the need for physical prototyping and excessive simulations, AI-driven material prediction 
            significantly cuts down resource consumption and environmental waste.
        """, style={"marginLeft": "20px"}),

        html.Li(html.B("⚡ Computational Efficiency")),
        html.P("""
            Machine learning models deliver results in seconds, bypassing the heavy cost and time of physics-based 
            simulations like DFT — enabling faster R&D cycles.
        """, style={"marginLeft": "20px"}),

        html.Li(html.B("🏭 Manufacturing Advantages")),
        html.P("""
            Predictive modeling streamlines material selection, reduces defects, and boosts yield — improving both 
            product quality and operational efficiency.
        """, style={"marginLeft": "20px"}),

        html.Li(html.B("🏢 Industry Innovation")),
        html.P("""
            Semiconductor companies can use these tools to develop next-generation chips that are faster, smaller, 
            and more energy-efficient — giving them a competitive edge in a rapidly evolving market.
        """, style={"marginLeft": "20px"})
    ])
], style={"padding": "20px 10px", "maxWidth": "1100px", "margin": "auto"})

