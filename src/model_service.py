from pathlib import Path

import joblib
import pandas as pd


class ModelService:

    FEATURES = [
        "Type_H",
        "Type_L",
        "Type_M",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]

    def __init__(self):
        base_dir = Path(__file__).resolve().parent.parent
        model_dir = base_dir / "models"

        self.model = joblib.load(
            model_dir / "gradient_boosting_model.pkl"
        )

        self.scaler = joblib.load(
            model_dir / "scaler.pkl"
        )

    def prepare_input(
        self,
        machine_type: str,
        air_temperature: float,
        process_temperature: float,
        rotational_speed: float,
        torque: float,
        tool_wear: float,
    ) -> pd.DataFrame:

        return pd.DataFrame([{
            "Type_H": int(machine_type == "H"),
            "Type_L": int(machine_type == "L"),
            "Type_M": int(machine_type == "M"),
            "Air temperature [K]": air_temperature,
            "Process temperature [K]": process_temperature,
            "Rotational speed [rpm]": rotational_speed,
            "Torque [Nm]": torque,
            "Tool wear [min]": tool_wear,
        }], columns=self.FEATURES)

    def predict(
        self,
        machine_type: str,
        air_temperature: float,
        process_temperature: float,
        rotational_speed: float,
        torque: float,
        tool_wear: float,
    ) -> tuple[int, float]:

        features = self.prepare_input(
            machine_type,
            air_temperature,
            process_temperature,
            rotational_speed,
            torque,
            tool_wear,
        )

        scaled_features = self.scaler.transform(features.to_numpy())

        scaled_features = pd.DataFrame(
            scaled_features,
            columns=self.FEATURES,
)

        prediction = int(self.model.predict(scaled_features)[0])
        probability = float(
            self.model.predict_proba(scaled_features)[0, 1]
)
        return prediction, round(probability, 4)