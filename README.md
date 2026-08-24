# Predictive Maintenance Platform

A machine-learning-powered platform for predicting industrial equipment
failures from machine and process measurements.

The project includes data analysis, preprocessing, machine-learning model
development, a prediction pipeline, and a FastAPI-based prediction API.

## Problem

Unexpected equipment failures can lead to production downtime, increased
maintenance costs, and operational disruptions.

This project aims to predict machine failure risk using machine and process
measurements such as temperature, rotational speed, torque, and tool wear.

## Dataset

This project uses the **AI4I 2020 Predictive Maintenance Dataset** from the
UCI Machine Learning Repository.

The dataset contains 10,000 observations and includes machine and process
measurements such as:

- Air temperature
- Process temperature
- Rotational speed
- Torque
- Tool wear
- Machine type

The dataset also contains machine failure labels and failure-mode indicators.

The dataset is synthetic but was designed to reflect predictive maintenance
data encountered in industrial environments.

**Source:** UCI Machine Learning Repository  
**Dataset:** AI4I 2020 Predictive Maintenance Dataset  
**DOI:** 10.24432/C5HS5C  
**License:** CC BY 4.0

## Project Structure

```text
predictive-maintenance-platform/
│
├── data/
│   ├── raw/
│   │   └── ai4i2020.csv
│   └── processed/
│
├── models/
│   ├── gradient_boosting_model.pkl
│   └── scaler.pkl
│
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_model_development.ipynb
│   └── 04_prediction_pipeline.ipynb
│
├── src/
│   ├── api.py
│   └── model_service.py
│
├── tests/
│   └── test_api.py
│
├── .gitignore
├── requirements.txt
└── README.md