# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

class CropInput(BaseModel):
    N: int
    P: int
    K: int
    temperature: float
    humidity: float
    ph: float
    rainfall: float

@app.post("/predict")
def predict_crop(data: CropInput):
    input_data = [[
        data.N, data.P, data.K,
        data.temperature, data.humidity,
        data.ph, data.rainfall
    ]]
    model = joblib.load("crop_model.pkl")
    prediction = model.predict(input_data)[0]
    return {"recommended_crop": prediction}
