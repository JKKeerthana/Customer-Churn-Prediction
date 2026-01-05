# Customer Churn Analysis & Prediction
## Overview

This project analyzes customer churn patterns and builds a machine learning model to predict churn probability. The goal is to identify key drivers of churn and support data-driven retention strategies for telecom businesses.
An interactive web application is included to predict churn for individual customers, built with Python, Streamlit, and a Random Forest model.

## Dataset
Source: Telco Customer Churn Dataset

Size: 7,043 customers

Features: 21 features including contract type, tenure, services, and billing details

## Pipeline Flow
### Data Preprocessing

Raw data is cleaned and transformed for modeling.

Key preprocessing steps:

Handle missing values

Convert categorical variables into numeric formats

Feature engineering (e.g., total charges calculation)

Run the preprocessing script:
python notebooks/01_data_cleaning.py

Output: cleaned_telco_churn.csv (saved in data/processed)

### Training Pipeline

A Random Forest Classifier is trained on the processed data.

Steps include:

Encoding categorical variables

Train/test split

Model training and evaluation

Saving the trained model

Run the training script:
python notebooks/04_churn_model.py


#### Outputs:

Model file: models/churn_model.pkl

Accuracy: ~78%

ROC-AUC: ~0.82

### Exploratory Data Analysis (EDA)

Visualize patterns in customer churn:

Churn by contract type

Churn by tenure groups

Monthly charges vs churn

Run the EDA script:
python notebooks/03_eda_analysis.py

### Web Application

Predict churn probability for individual customers.

Features:

Interactive customer profile input

Clean UI with background colors and cards

Displays prediction metrics and business insights

Run the app locally:
streamlit run app.py


