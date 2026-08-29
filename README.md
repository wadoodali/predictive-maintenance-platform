# Predictive Maintenance Platform

A machine-learning-powered platform for predicting industrial equipment failures from machine and process measurements.

The project includes data analysis, preprocessing, machine-learning model development, a prediction pipeline, a FastAPI-based prediction API, and an interactive Streamlit dashboard.

## Problem

Unexpected equipment failures can lead to production downtime, increased maintenance costs, and operational disruptions.

This project aims to predict machine failure risk using machine and process measurements such as temperature, rotational speed, torque, and tool wear.

## Dataset

This project uses the **AI4I 2020 Predictive Maintenance Dataset** from the UCI Machine Learning Repository.

The dataset contains 10,000 observations and includes machine and process measurements such as:

- Air temperature
- Process temperature
- Rotational speed
- Torque
- Tool wear
- Machine type

The dataset also contains machine failure labels and failure-mode indicators.

The dataset is synthetic but was designed to reflect predictive maintenance data encountered in industrial environments.

**Source:** UCI Machine Learning Repository  
**Dataset:** AI4I 2020 Predictive Maintenance Dataset  
**DOI:** 10.24432/C5HS5C  
**License:** CC BY 4.0

## Machine Learning Model

A **Gradient Boosting Classifier** was selected as the final machine-learning model for predicting machine failures.

The model was trained on the preprocessed training data and evaluated on a held-out test set.

## Model Performance

The final model achieved the following results on the test set:

| Metric | Score |
|---|---:|
| Accuracy | 98.50% |
| Precision | 80.65% |
| Recall | 73.53% |
| F1-Score | 76.92% |
| ROC-AUC | 96.97% |

The dataset is highly imbalanced, with significantly more non-failure cases than failure cases. Therefore, accuracy alone is not sufficient for evaluating the model.

Recall is particularly important in predictive maintenance because failing to identify an actual machine failure can result in unexpected downtime and maintenance costs.

The model achieved a **73.53% recall for the failure class**, while maintaining a **96.97% ROC-AUC**, indicating strong overall discrimination between failure and non-failure cases.

## Prediction API

The trained model is integrated into a **FastAPI-based prediction API**.

The API accepts machine and process measurements and returns:

- Failure prediction
- Failure probability
- Prediction message
- Feature importance values

The API validates incoming inputs using Pydantic and provides an interactive Swagger documentation interface.

### API Endpoint

```text
POST /predict

**### Example Request**

\`\`\`json
{
  "machine_type": "L",
  "air_temperature": 298.1,
  "process_temperature": 308.6,
  "rotational_speed": 1551,
  "torque": 42.8,
  "tool_wear": 0
}
\`\`\`

**### Example Response**

\`\`\`json
{
  "prediction": 0,
  "failure_probability": 0.0072,
  "message": "No machine failure predicted",
  "feature_importance": {
    "Machine Type": 0.4017302435729959,
    "Air Temperature": 0.40177153030205787,
    "Process Temperature": 0.1745267249020629,
    "Rotational Speed": 0.0004193705931542025,
    "Torque": 0.01849229590937616,
    "Tool Wear": 0.0030598347203531135
  }
}
\`\`\`

The prediction threshold is set to **0.35**. A failure probability greater than or equal to this threshold is classified as a predicted machine failure.

The API functionality is covered by automated tests using `pytest`.

**## Interactive Dashboard**

The project includes an interactive **Streamlit dashboard** that allows users to enter current machine operating conditions and receive a failure-risk assessment.

Users can provide:

\- Machine type

\- Air temperature

\- Process temperature

\- Rotational speed

\- Torque

\- Tool wear

The dashboard displays:

\- Estimated machine health

\- Failure probability

\- Predicted failure status

\- Risk level

\- Recommended maintenance action

\- Key model drivers

\- Current operating conditions

The dashboard communicates with the FastAPI backend for predictions, ensuring that the same deployed model is used for dashboard predictions and API requests.

**## Docker**

The application is containerized using **Docker** and **Docker Compose**.

The Docker setup consists of two services:

\- **API:** FastAPI prediction service

\- **Dashboard:** Streamlit interactive dashboard

Both services are built from the project's Dockerfile.

**### Start the Application**

Make sure Docker Desktop is running, then execute:

\`\`\`bash
docker compose up -d --build
\`\`\`

Check the running containers:

\`\`\`bash
docker compose ps
\`\`\`

**### Access the Application**

FastAPI:

\`\`\`text
http://localhost:8000
\`\`\`

FastAPI Swagger documentation:

\`\`\`text
http://localhost:8000/docs
\`\`\`

Streamlit dashboard:

\`\`\`text
http://localhost:8501
\`\`\`

**### Stop the Application**

\`\`\`bash
docker compose down
\`\`\`

**## How to Run Locally**

**### 1. Clone the Repository**

\`\`\`bash
git clone https://github.com/wadoodali/predictive-maintenance-platform.git
cd predictive-maintenance-platform
\`\`\`

**### 2. Create a Virtual Environment**

\`\`\`bash
python -m venv .venv
\`\`\`

Activate it on Windows:

\`\`\`bash
.venv\Scripts\activate
\`\`\`

**### 3. Install Dependencies**

\`\`\`bash
pip install -r requirements.txt
\`\`\`

**### 4. Start the FastAPI Server**

\`\`\`bash
uvicorn src.api:app --reload --port 8000
\`\`\`

Keep this terminal running.

**### 5. Start the Streamlit Dashboard**

Open a second terminal in the project folder.

Activate the virtual environment:

\`\`\`bash
.venv\Scripts\activate
\`\`\`

Then run:

\`\`\`bash
streamlit run app.py
\`\`\`

The Streamlit dashboard will open in your browser.

**### 6. Run the Tests**

To run the automated API tests:

\`\`\`bash
python -m pytest
\`\`\`

The test suite should complete successfully.

**## Project Structure**

\`\`\`text
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
├── .dockerignore
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
\`\`\`

**## Technologies**

\- Python

\- Pandas

\- NumPy

\- Scikit-learn

\- Joblib

\- FastAPI

\- Pydantic

\- Streamlit

\- Pytest

\- Docker

\- Docker Compose

\- Jupyter Notebook

**## Key Features**

\- Exploratory data analysis

\- Data preprocessing

\- Machine-learning model development

\- Gradient Boosting classification

\- Failure probability prediction

\- Configurable classification threshold

\- FastAPI prediction service

\- Input validation using Pydantic

\- Automated API testing

\- Interactive Streamlit dashboard

\- Feature importance analysis

\- Dockerized application

\- Docker Compose orchestration