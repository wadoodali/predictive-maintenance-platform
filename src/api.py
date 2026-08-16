from fastapi import FastAPI
from pydantic import BaseModel
import joblib
from pathlib import Path

app = FastAPI(title="Predictive Maintenance API")

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

model = joblib.load(MODEL_DIR / "gradient_boosting_model.pkl")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")


class MachineInput(BaseModel):
    machine_type: str
    air_temperature: float
    process_temperature: float
    rotational_speed: float
    torque: float
    tool_wear: float


@app.get("/")
def home():
    return {"message": "Predictive Maintenance API is running"}


@app.post("/predict")
def predict(machine: MachineInput):
    return {"message": "Prediction endpoint is working"}