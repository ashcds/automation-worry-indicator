# 🤖 Automation Worry Predictor  
*A Machine Learning Model to Identify Individuals Concerned About Job Automation*

Try out the [live App!](https://automation-worry-indicator.streamlit.app/)

## 📘 Overview

This project builds an end-to-end machine learning pipeline to predict whether a person is **worried about automation replacing human jobs**, using survey data from **Pew Research Center's Wave 27 American Trends Panel**.

By combining statistical feature selection, model tuning, class balancing, and SHAP-based explainability, this project offers insights into **who is most concerned about automation** and why. This can provide valuable and actionable insights for employers, policymakers and workforce development programs.


## 🎯 Objectives

- Predict concern about job automation (`Target`)
- Explore key factors such as job type, beliefs about AI, and demographics
- Build interpretable and accurate ML models
- Apply best practices in feature selection, cross-validation, and hyperparameter tuning

## 📊 Dataset

- **Source**: [Pew Research Center — Wave 27](https://www.pewresearch.org/)
- **Size**: ~2,500 respondents, 250+ features
- **Target Variable**: `ROBJOB3B_W27` → binary class (`Worried` vs `Not Worried`)

## 🧪 Methodology

### ✅ Data Processing
- Replaced special codes (e.g., 99 = "Don’t know") with `NaN`
- Engineered the `Target` variable
- Removed features with excessive missing values 
- Used **RandomOverSampler** to balance the dataset

### ✅ Feature Selection
- **Chi-squared test** for statistical relevance
- **SHAP values** from Random Forest for model-driven importance
- Final feature set based on overlap and consistency between methods

### ✅ Models Trained
- Logistic Regression
- Random Forest (tuned)
- XGBoost (tuned)

### ✅ Model Evaluation
- Cross-validation (`cv=5`) using F1 score
- Confusion matrix, precision, recall, ROC AUC
- SHAP summary plots for interpretability

## 🏆 Results

- **Best Model**: Tuned XGBoost
- **F1 Score**: 0.804  
- **Accuracy**: 73%  
- **Recall (Worried class)**: 80%  
- **Top Features**: Concerns about societal impact, job loss likelihood, driverless cars
