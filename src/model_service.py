from pathlib import Path

import joblib
import pandas as pd


class ModelService:

    SCALER_FEATURES = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
        "Type_H",
        "Type_L",
        "Type_M",
    ]

    MODEL_FEATURES = [
        "Type_H",
        "Type_L",
        "Type_M",
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]

    THRESHOLD = 0.35

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

        machine = pd.DataFrame(
            [{
                "Air temperature [K]": air_temperature,
                "Process temperature [K]": process_temperature,
                "Rotational speed [rpm]": rotational_speed,
                "Torque [Nm]": torque,
                "Tool wear [min]": tool_wear,
                "Type_H": int(machine_type == "H"),
                "Type_L": int(machine_type == "L"),
                "Type_M": int(machine_type == "M"),
            }],
            columns=self.SCALER_FEATURES,
        )

        # Apply the same scaling used during model development
        scaled_machine = self.scaler.transform(machine)

        scaled_machine = pd.DataFrame(
            scaled_machine,
            columns=self.SCALER_FEATURES,
        )

        # Apply the same feature order used by the model
        scaled_machine = scaled_machine[self.MODEL_FEATURES]

        return scaled_machine

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

        probability = float(
            self.model.predict_proba(features)[0, 1]
        )

        prediction = int(
            probability >= self.THRESHOLD
        )

        return prediction, round(probability, 4)

    def get_feature_importance(self) -> dict[str, float]:

        importances = self.model.feature_importances_

        return {
            "Machine Type": float(
                importances[0]
                + importances[1]
                + importances[2]
            ),
            "Air Temperature": float(importances[3]),
            "Process Temperature": float(importances[4]),
            "Rotational Speed": float(importances[5]),
            "Torque": float(importances[6]),
            "Tool Wear": float(importances[7]),
        }