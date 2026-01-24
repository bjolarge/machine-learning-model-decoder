from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import traceback

app = FastAPI()

#LOAD MODEL (once at startup)
try:
    model = joblib.load("india_air_quality.joblib")
except Exception as e:
    print("Model failed to load")
    traceback.print_exc()
    raise e


# INPUT SCHEMA
class AirQualityInput(BaseModel):
    PM25: float
    PM10: float
    NO: float
    NO2: float
    NOx: float
    NH3: float
    CO: float
    SO2: float
    O3: float
    Benzene: float
    Toluene: float
    Xylene: float


# 🔹 PREDICTION ENDPOINT
@app.post("/predict")
def predict(data: AirQualityInput):
    try:
        features = np.array([[
            data.PM25,
            data.PM10,
            data.NO,
            data.NO2,
            data.NOx,
            data.NH3,
            data.CO,
            data.SO2,
            data.O3,
            data.Benzene,
            data.Toluene,
            data.Xylene
        ]], dtype=np.float32)

        prediction = model.predict(features)

        # TensorFlow / NumPy safety
        if hasattr(prediction, "numpy"):
            prediction = prediction.numpy()

        return {
            "prediction": prediction.tolist()
        }

    except Exception as e:
        print("🔥 Prediction failed")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))



# from fastapi import FastAPI,Request
# import joblib
# import numpy as np

# app = FastAPI()
# model = joblib.load("air_quality_model.joblib")

# @app.get("/")
# async def home(request: Request):
#     return {"status": "ML API is running"}

# @app.post("/predict")
# def predict(data: dict):
#     features = np.array([[
#         data["NO2 AQI Value"],
#         data["Ozone AQI Value"],
#         data["CO AQI Value"],
#         data["PM2.5 AQI Value"]
#     ]])

#     prediction = model.predict(features)[0]

#     return {
#         "aqi": float(prediction),
#         "risk_level": (
#             "high" if prediction > 150 else
#             "moderate" if prediction > 80 else
#             "low"
#         )
#     }



# from fastapi import FastAPI, Request
# import joblib
# import numpy as np

# app = FastAPI()

# # Load model once at startup
# model = joblib.load("water.joblib")

# # -----------------------------------
# # pH rule-based logic
# # -----------------------------------
# def classify_water_by_ph(ph):
#     if ph < 6.5:
#         return "Acidic"
#     elif 6.5 <= ph <= 7.5:
#         return "Safe for drinking"
#     else:
#         return "Basic"

# @app.get("/")
# async def home(request: Request):
#     return {"status": "Water Quality ML API is running"}

# @app.post("/predict")
# def predict(data: dict):
#     features = np.array([[ 
#         data["PH"],
#         data["Conductivity"],
#         data["BOD"],
#         data["Suspended Solids"],
#         data["Total Oxidized N"],
#         data["Coliform Bacteria"]
#     ]])

#     prediction = model.predict(features)[0]

#     ph_status = classify_water_by_ph(data["PH"])

#     return {
#         "ph": data["PH"],
#         "ph_status": ph_status,
#         "model_prediction": float(prediction),
#         "final_water_status": (
#             "Safe for drinking"
#             if ph_status == "Safe for drinking"
#             else "Unsafe"
#         )
#     }

