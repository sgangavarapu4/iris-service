import os
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import aiplatform

# --- CONFIGURATION ---
PROJECT_ID = "project-73119d0a-ddcc-44c1-8f3"
REGION = "us-central1"
BUCKET = f"gs://{PROJECT_ID}-artifacts"

# Helper to find files inside the 'app' folder in Docker
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_FILE = os.path.join(BASE_DIR, "iris_pipeline.yaml")
MODEL_PATH = os.path.join(BASE_DIR, "iris_model_v1.pkl")

# Initialize Vertex AI SDK
aiplatform.init(project=PROJECT_ID, location=REGION)

app = FastAPI(title="Iris Hybrid Service")

class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

SPECIES_MAP = {0: "setosa", 1: "versicolor", 2: "virginica"}

# Load local model for inference
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    print(f"Model load failed: {e}")
    model = None

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": bool(model)}

# --- Training Trigger ---
@app.post("/train")
async def trigger_training():
    if not os.path.exists(PIPELINE_FILE):
        raise HTTPException(status_code=404, detail=f"Pipeline file not found at {PIPELINE_FILE}")
        
    try:
        job = aiplatform.PipelineJob(
            display_name="iris-manual-retrain",
            template_path=PIPELINE_FILE,
            pipeline_root=f"{BUCKET}/pipeline_root",
            parameter_values={
                "project_id": PROJECT_ID,
                "bucket_name": BUCKET.replace("gs://", "") 
            }
        )
        job.submit()
        return {
            "status": "success", 
            "job_id": job.name, 
            "detail": "Vertex AI Pipeline submitted"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Inference ---
@app.post("/predict")
async def predict(data: IrisInput):
    if model is None:
        raise HTTPException(status_code=500, detail="Inference engine unavailable")
    
    features = [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
    prediction_idx = int(model.predict(features)[0])
    return {
        "species": SPECIES_MAP.get(prediction_idx),
        "prediction_index": prediction_idx
    }