import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

# 1. Load the model globally (Pre-loading)
# This ensures the .pkl is in RAM and ready for instant requests
MODEL_PATH = "iris_model_v1.pkl"

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

app = FastAPI(title="Iris Prediction Service")

# 2. Data Validation Schema
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# 3. Species Mapping
SPECIES_MAP = {
    0: "setosa",
    1: "versicolor",
    2: "virginica"
}

# --- ENDPOINTS ---

@app.get("/health")
async def health_check():
    """Endpoint for Docker/Kubernetes health monitoring."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_version": MODEL_PATH}

@app.post("/predict")
async def predict(data: IrisInput):
    """Main inference endpoint."""
    if model is None:
        raise HTTPException(status_code=500, detail="Inference engine unavailable")

    # Convert input to the 2D array format sklearn expects
    features = [[
        data.sepal_length, 
        data.sepal_width, 
        data.petal_length, 
        data.petal_width
    ]]
    
    prediction_idx = int(model.predict(features)[0])
    
    return {
        "prediction": prediction_idx,
        "species": SPECIES_MAP.get(prediction_idx, "unknown"),
        "model_version": MODEL_PATH
    }