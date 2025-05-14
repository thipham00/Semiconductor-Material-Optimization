# Semiconductor Material Optimization

This project uses machine learning (ML) and deep learning (DL) techniques to predict the **band gap** of semiconductor materials — a key property that determines electrical conductivity. By leveraging simulation data from ORCA (using Density Functional Theory, DFT) and structural data from the Materials Project, we built predictive models that significantly reduce the time and cost of materials discovery. The final result is a fully interactive web dashboard for visualizing model performance, feature importance, and computational efficiency.

---

## Setup Instructions
### 1. Clone the Repository
```
git clone https://github.com/thipham00/Semiconductor-Material-Optimization.git  
cd semiconductor-material-optimization
```  

### 2. Create a Virtual Environment (optional)
``` 
python -m venv venv  
source venv/bin/activate             #On Windows: venv\Scripts\activate
```  
### 3. Install Required Packages (Make sure to have Python 3.8+ installed.)
Run the following command to install the required packages:  
```pip install -r requirements.txt```  


Or manually install the following packages:  
- scikit-learn  
- xgboost  
- pandas  
- numpy  
- matplotlib  
- seaborn  
- networkx  
- pytorch  

## Project Pipeline Overview
[1] Data Collection  
↓  
[2] Data Cleaning & Feature Engineering  
↓  
[3] Model Training (ML & GNNs)  
↓  
[4] Model Evaluation (MSE, RMSE, R²)  
↓  
[5] Visualization & Analysis  
↓  
[6] Dashboard Deployment  

#### Step 1: Data Collection
- Materials Project API: Used to extract structural and atomic data (e.g., crystal system, space group, density, formula).
- ORCA DFT Simulations: Generated accurate band gap values and other quantum descriptors for training labels.

#### Step 2: Data Cleaning & Feature Engineering
- Dropped nulls and duplicates
- Extracted weighted atomic properties using pymatgen
- Encoded categorical features and standardized numerical ones

#### Step 3: Model Training
- Traditional: Linear Regression, Random Forest, XGBoost
- Deep Learning: Fully Connected Neural Network (FCNN)
- Graph-Based: Graph Neural Network (GNN) using PyTorch Geometric

#### Step 4: Model Evaluation
- Evaluated using MSE, RMSE, and R²
- Compared with baseline ORCA accuracy and computation time

#### Step 5: Visualization
- Model performance bar charts
- Feature correlation heatmaps
- Residual scatter plots
- Timing comparison (ORCA vs ML)

#### Step 6: Dashboard Deployment
- Built with Dash + Plotly
- Organized into modular pages with interactive graphs

## Repository Structure
```
.
├── app.py             #Main Dash app launcher
├── app.yaml           #Configuration file (e.g., for deployment)
├── assets             #Folder containing images, CSS, and other static files
│   ├── after-cleaning.png
│   ├── bandgap-distribution.png
│   ├── before-cleaning.png
│   ├── correlation-heatmap.png
│   ├── features-correlation-w-bandgap.png
│   ├── mp-logo.png
│   ├── orca-logo.png
│   ├── styles.css
│   └── true-predicted (RF).png
├── data               #Datasets
│   ├── mp_summary_selected_columns.csv
│   └── mp_summary_with_weighted_atomic_properties.csv
├── LICENSE            #License page
├── notebooks          #Jupyter notebooks used for data exploration/analysis/training models
│   └── cs163 (1).ipynb
├── pages              #Modular Dash app pages
│   ├── findings.py    #Model results and visualizations
│   ├── main.py        #Homepage with project overview
│   ├── methods.py     #Data processing, modeling, and methods used in the project
│   └── objective.py   #Project goals and data sources
├── README.md          #Project documentation
└── requirements.txt   #All Python dependencies for the project
```
## Website
https://carbon-caster-451220-t5.uc.r.appspot.com/

## References
- 🔗 [Liu et al., 2023 – Accurate prediction of semiconductor bandgaps using ML](https://doi.org/10.1016/j.mtcomm.2023.106578)  
  *Materials Today Communications, Volume 36, August 2023, Article 106578*


## Contributors
Thi Dao Nguyen Pham — thidaonguyen.pham@gmail.com  
Jewon Yeon — yeon971105@gmail.com