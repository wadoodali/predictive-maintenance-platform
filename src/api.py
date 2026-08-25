from typing import Literal
from fastapi import FastAPI
from pydantic import BaseModel, Field
from src.model_service import ModelService

app = FastAPI(title="Predictive Maintenance API")
model_service = ModelService()

class MachineInput(BaseModel):
    machine_type: Literal["L", "M", "H"]
    air_temperature: float = Field(..., ge=250, le=350)
    process_temperature: float = Field(..., ge=250, le=350)
    rotational_speed: float = Field(..., ge=0, le=5000)
    torque: float = Field(..., ge=0, le=100)
    tool_wear: float = Field(..., ge=0, le=300)

class PredictionResponse(BaseModel):
    prediction: int
    failure_probability: float
    message: str
    feature_importance: dict[str, float]

@app.get("/")
def home():
    return {"message": "Predictive Maintenance API is running"}

@app.post("/predict", response_model=PredictionResponse)
def predict(machine: MachineInput):
    prediction, probability = model_service.predict(
        machine_type=machine.machine_type,
        air_temperature=machine.air_temperature,
        process_temperature=machine.process_temperature,
        rotational_speed=machine.rotational_speed,
        torque=machine.torque,
        tool_wear=machine.tool_wear,
    )

    feature_importance = model_service.get_feature_importance()

    return PredictionResponse(
        prediction=prediction,
        failure_probability=probability,
        message=(
            "Machine failure predicted"
            if prediction
            else "No machine failure predicted"
        ),
        feature_importance=feature_importance,
    )