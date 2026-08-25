# Predictive Maintenance Platform

A machine-learning-powered platform for predicting industrial equipment
failures from machine and process measurements.

The project includes data analysis, preprocessing, machine-learning model
development, a prediction pipeline, a FastAPI-based prediction API, and an
interactive Streamlit dashboard.

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

## Machine Learning Model

A **Gradient Boosting Classifier** was selected as the final machine-learning
model for predicting machine failures.

The model was trained on the preprocessed training data and evaluated on a
held-out test set.

## Model Performance

The final model achieved the following results on the test set:

| Metric | Score |
|---|---:|
| Accuracy | 98.55% |
| Precision | 88.24% |
| Recall | 66.18% |
| F1-Score | 75.63% |
| ROC-AUC | 96.97% |

The dataset is highly imbalanced, with significantly more non-failure cases
than failure cases. Therefore, accuracy alone is not sufficient for evaluating
the model.

Recall is particularly important in predictive maintenance because failing
to identify an actual machine failure can result in unexpected downtime and
maintenance costs.

The model achieved a **66.18% recall for the failure class**, while maintaining
a **96.97% ROC-AUC**, indicating strong overall discrimination between failure
and non-failure cases.

## Prediction API

The trained model is integrated into a **FastAPI-based prediction API**.

The API accepts machine and process measurements and returns a predicted
machine failure result.

The API functionality is covered by automated tests using `pytest`.

## Interactive Dashboard

The project includes an interactive **Streamlit dashboard** that allows users
to enter current machine operating conditions and receive a failure-risk
assessment.

Users can provide:

- Machine type
- Air temperature
- Process temperature
- Rotational speed
- Torque
- Tool wear

The dashboard displays:

- Estimated machine health
- Failure probability
- Predicted failure status
- Risk level
- Recommended maintenance action
- Key model drivers
- Current operating conditions

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/wadoodali/predictive-maintenance-platform.git
cd predictive-maintenance-platform
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the FastAPI server

```bash
uvicorn src.api:app --reload --port 8000
```

Keep this terminal running.

### 5. Start the Streamlit dashboard

Open a second terminal in the project folder.

Activate the virtual environment:

```bash
.venv\Scripts\activate
```

Then run:

```bash
streamlit run app.py
```

The Streamlit dashboard will open in your browser.

### 6. Run the tests

To run the automated API tests:

```bash
python -m pytest
```

The test suite should complete successfully.

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
│   ├── __init__.py
│   ├── api.py
│   └── model_service.py
│
├── tests/
│   └── test_api.py
│
├── app.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- FastAPI
- Streamlit
- Pytest
- Jupyter Notebook