from fastapi import FastAPI, Request
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("water.joblib")

def classify_water_by_ph(ph):
    if ph < 6.5:
        return "Acidic"
    elif 6.5 <= ph <= 7.5:
        return "Safe for drinking"
    else:
        return "Basic"

@app.get("/")
async def home(request: Request):
    return {"status": "Water Quality ML API is running"}

@app.post("/predict")
def predict(data: dict):
    features = np.array([[ 
        data["PH"],
        data["Conductivity"],
        data["BOD"],
        data["Suspended Solids"],
        data["Total Oxidized N"],
        data["Coliform Bacteria"]
    ]])

    prediction = model.predict(features)[0]

    ph_status = classify_water_by_ph(data["PH"])

    return {
        "ph": data["PH"],
        "ph_status": ph_status,
        "model_prediction": float(prediction),
        "final_water_status": (
            "Safe for drinking"
            if ph_status == "Safe for drinking"
            else "Unsafe"
        )
    }