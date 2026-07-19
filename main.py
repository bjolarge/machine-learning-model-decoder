# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import numpy as np
# import joblib
# import traceback
# from water import app as water_app;
# from fuelefficiency import app as fuel_app;

# app = FastAPI()

# app.mount("/water", water_app)
# app.mount("/fuel", fuel_app)


# try:
#     model = joblib.load("india_air_quality.joblib")
# except Exception as e:
#     print("Model failed to load")
#     traceback.print_exc()
#     raise e


# # INPUT SCHEMA
# class AirQualityInput(BaseModel):
#     PM25: float
#     PM10: float
#     NO: float
#     NO2: float
#     NOx: float
#     NH3: float
#     CO: float
#     SO2: float
#     O3: float
#     Benzene: float
#     Toluene: float
#     Xylene: float


# # 🔹 PREDICTION ENDPOINT
# @app.post("/predict")
# def predict(data: AirQualityInput):
#     try:
#         features = np.array([[
#             data.PM25,
#             data.PM10,
#             data.NO,
#             data.NO2,
#             data.NOx,
#             data.NH3,
#             data.CO,
#             data.SO2,
#             data.O3,
#             data.Benzene,
#             data.Toluene,
#             data.Xylene
#         ]], dtype=np.float32)

#         prediction = model.predict(features)

#         # TensorFlow / NumPy safety
#         if hasattr(prediction, "numpy"):
#             prediction = prediction.numpy()

#         return {
#             "prediction": prediction.tolist()
#         }

#     except Exception as e:
#         print("🔥 Prediction failed")
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=str(e))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import joblib
import traceback
from water import app as water_app
from fuelefficiency import app as fuel_app

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://machinelearningfrontendenvironmenta.vercel.app",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/water", water_app)
app.mount("/fuel", fuel_app)


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


@app.get("/")
def home():
    return {"status": "ML API is running"}


# 🔹 PREDICTION ENDPOINT
# @app.post("/predict")
# def predict(data: AirQualityInput):
#     try:
#         features = np.array([[
#             data.PM25,
#             data.PM10,
#             data.NO,
#             data.NO2,
#             data.NOx,
#             data.NH3,
#             data.CO,
#             data.SO2,
#             data.O3,
#             data.Benzene,
#             data.Toluene,
#             data.Xylene
#         ]], dtype=np.float32)

#         prediction = model.predict(features)

#         if hasattr(prediction, "numpy"):
#             prediction = prediction.numpy()

#         return {
#             "prediction": prediction.tolist()
#         }

#     except Exception as e:
#         print("Prediction failed")
#         traceback.print_exc()
#         raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
def predict(data: AirQualityInput):
    try:
        features = np.array([[
            data.PM25, data.PM10, data.NO, data.NO2, data.NOx,
            data.NH3, data.CO, data.SO2, data.O3,
            data.Benzene, data.Toluene, data.Xylene
        ]], dtype=np.float32)

        prediction = model.predict(features)

        if hasattr(prediction, "numpy"):
            prediction = prediction.numpy()

        aqi_value = float(np.ravel(prediction)[0])
        classification = classify_air_quality(aqi_value)

        return {
            "aqi": aqi_value,
            "category": classification["category"],
            "is_safe": classification["is_safe"]
        }

    except Exception as e:
        print("Prediction failed")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))