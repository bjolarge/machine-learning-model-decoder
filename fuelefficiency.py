from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from typing import List

app = FastAPI(
    title="Fuel Efficiency Prediction API",
    version="1.0.0"
)

model = joblib.load("fuel_efficiency_model.joblib")


class FuelInput(BaseModel):
    cylinders: float
    displacement: float
    horsepower: float
    weight: float
    acceleration: float
    model_year: float

def classify_efficiency(mpg: float) -> str:
    if mpg < 15:
        return "very fuel inefficient"
    elif mpg < 25:
        return "average fuel"
    elif mpg < 35:
        return "good fuel"
    elif mpg < 50:
        return "very fuel efficient"
    else:
        return "extremely fuel efficient"

@app.get("/")
def home():
    return {"status": "Fuel Efficiency Joblib Model API running"}


# @app.post("/predict")
# def predict(data: FuelInput):
#     features = [
#         data.cylinders,
#         data.displacement,
#         data.horsepower,
#         data.weight,
#         data.acceleration,
#         data.model_year
#     ]
#     X = np.array(features, dtype=np.float32).reshape(1, -1)
#     prediction = model.predict(X)
#     return {"fuel_efficiency_mpg": float(prediction[0])}


@app.post("/predict")
def predict(data: FuelInput):
    features = [
        data.cylinders,
        data.displacement,
        data.horsepower,
        data.weight,
        data.acceleration,
        data.model_year
    ]
    X = np.array(features, dtype=np.float32).reshape(1, -1)
    prediction = model.predict(X)[0]
    
    efficiency_label = classify_efficiency(prediction)
    
    return {
        "fuel_efficiency_mpg": float(prediction),
        "efficiency": efficiency_label
    }
