# LightGBM-Carbon-Flux-Prediction

This repository contains two Python scripts designed to predict carbon fluxes using machine learning models trained on data from Eddy Covariance sites across Europe.

## Model Variants

- `non-NEE-derived_model.py`:  
  A version that excludes any features derived from the target variable (NEE), intended for use as a standalone predictive tool when NEE is not available.

- `NEE-derived_model.py`:  
  A version that includes lag and rolling features of NEE, providing stronger predictive performance for use as a **gap-filling tool** alongside Eddy Covariance Towers.

## Dependencies

The following Python libraries are required:

- pandas  
- numpy  
- matplotlib  
- seaborn  
- lightgbm  
- optuna  
- scikit-learn  

You can install them using pip:

```bash
pip install pandas numpy matplotlib seaborn lightgbm optuna scikit-learn
