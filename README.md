
# Swiggy Delivery Time Prediction: Project Summary

## Overview
This project presents an end-to-end machine learning pipeline designed to predict food delivery times for Swiggy, one of India's leading on-demand food delivery platforms. The primary objective was to accurately estimate the delivery duration by analyzing operational, environmental, temporal, and rider-specific factors. The project follows a structured data science methodology, spanning from rigorous exploratory data analysis to complex model stacking and hyperparameter tuning.

## Methodology and Process

The project was executed through the following structured phases:

### 1. Data Cleaning and Imputation
The initial phase involved rigorous data inspection to identify and treat missing values, particularly in rider age, ratings, and temporal factors. Both imputation techniques (e.g., SimpleImputer, KNNImputer, IterativeImputer) and data-dropping strategies were evaluated using cross-validation. Empirical results demonstrated that dropping rows with missing values yielded a superior cross-validation score compared to imputation, establishing the foundation for the modeling dataset.

### 2. Exploratory Data Analysis (EDA)
A comprehensive univariate, bivariate, and multivariate analysis was conducted to understand the drivers of delivery time:
* **Target Variable:** The delivery time averaged ~26 minutes with a bimodal distribution.
* **Operational Factors:** Delivery distance (average 9 km) and multiple concurrent deliveries exhibited a strong, monotonic relationship with longer delivery times.
* **Environmental Factors:** Traffic conditions emerged as a primary driver, with median times ranging from ~20 minutes (no traffic) to ~30 minutes (heavy traffic/jams). Adverse weather and festival days also significantly extended delivery times.
* **Vehicle & Rider Characteristics:** Electric scooters consistently recorded the fastest delivery times across all traffic conditions. Rider ratings also showed statistical significance in predicting delivery efficiency.

### 3. Feature Selection
Feature importance was evaluated using Random Forest, Gradient Boosting Machine (GBM), and Recursive Feature Elimination (RFE). The analysis revealed that the top five features—rider ratings, weather, delivery distance, multiple deliveries, and rider age—accounted for approximately 70% of the combined feature importance. While dropping low-importance features was tested, the full feature set ultimately produced the most robust predictive performance, confirming that nuanced variables like vehicle condition and traffic capture vital variance.

### 4. Baseline Modeling and Hyperparameter Tuning
Various machine learning algorithms were trained and evaluated to establish strong baseline performances. The Optuna framework was heavily utilized for Bayesian hyperparameter optimization across several tree-based models:
* **Random Forest Regressor:** Tuned on the refined numerical dataset, achieving a robust Test Mean Absolute Error (MAE) of ~3.02 minutes and an R² score of 83.85%.
* **CatBoost Regressor:** Optimized using Optuna, showing strong performance with a Validation MAE of ~3.13 minutes.
* **LightGBM Regressor:** Demonstrated excellent learning capability, achieving a Validation MAE of ~3.16 minutes after hyperparameter tuning.
* **Artificial Neural Network (ANN):** A simple 1-layer neural network was also experimented with but yielded a higher validation MAE of ~4.68 minutes, making tree-based ensembles the preferred choice.

### 5. Meta-Modeling and Final Selection
To maximize predictive accuracy, the out-of-fold predictions from the optimized CatBoost, LightGBM, and Random Forest models were combined using a Stacking Regressor. A Lasso Regression model was employed as the meta-learner to intelligently weight the predictions from the base estimators. Optuna was once again utilized to tune the `alpha` penalty of the Lasso meta-model, resulting in a perfectly regularized fit with minimal gap between training and validation scores.

## Final Model and Results

The **Final Model** deployed for this project is the **Stacking Regressor** (utilizing CatBoost, LightGBM, and Random Forest as base estimators, and Lasso Regression as the meta-learner).

* **Test Mean Absolute Error (MAE):** 3.01 minutes
* **R² Score:** 83.9% (~84%)

This final architecture successfully reduced the prediction error to roughly 3 minutes, outperforming all individual baseline and meta-models. The final model pipeline, alongside its parameters and metrics, was successfully logged and versioned using MLflow and DagsHub to ensure total reproducibility and lifecycle management.
