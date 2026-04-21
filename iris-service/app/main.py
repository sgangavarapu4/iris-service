import os
import joblib
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import aiplatform, storage

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
PROJECT_ID = "project-73119d0a-ddcc-44c1-8f3"
REGION = "us-central1"
# Ensure the bucket name matches your actual GCS bucket
BUCKET_NAME = f"{PROJECT_ID}-artifacts"
GCS_MODEL_PATH = "models/iris_model_v1.pkl"

# Paths for local file handling
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_FILE = os.path.join(BASE_DIR, "iris_pipeline.yaml")
# We download the cloud model to /tmp to avoid permission issues in containers
LOCAL_DOWNLOAD_PATH = "/tmp/iris_model_v1.pkl"

# Initialize Vertex AI SDK
aiplatform.init(project=PROJECT_ID, location=REGION)

app = FastAPI(title="Iris Hybrid MLOps Service")

# --- DATA MODELS ---
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

SPECIES_MAP = {0: "setosa", 1: "versicolor", 2: "virginica"}

# --- MODEL LOADER ---
def load_latest_model():
    """Attempts to download the model from GCS, falls back to local if fails."""
    try:
        logger.info(f"Attempting to download model from gs://{BUCKET_NAME}/{GCS_MODEL_PATH}")
        storage_client = storage.Client()
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(GCS_MODEL_PATH)
        
        blob.download_to_filename(LOCAL_DOWNLOAD_PATH)
        logger.info("Successfully downloaded latest model from GCS.")
        return joblib.load(LOCAL_DOWNLOAD_PATH)
    
    except Exception as e:
        logger.warning(f"Could not load model from GCS: {e}")
        # Fallback to the model packaged inside the 'app' folder
        fallback_path = os.path.join(BASE_DIR, "iris_model_v1.pkl")
        if os.path.exists(fallback_path):
            logger.info("Loading fallback model from local directory.")
            return joblib.load(fallback_path)
        
        logger.error("No model found in GCS or local directory.")
        return None

# Global model variable
model = load_latest_model()

# --- ROUTES ---

@app.get("/health")
async def health():
    """Health check to verify model status."""
    return {
        "status": "healthy",
        "model_loaded": bool(model),
        "project_id": PROJECT_ID
    }

@app.post("/predict")
async def predict(data: IrisInput):
    """Predict iris species using the loaded model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Inference engine unavailable. Model not loaded.")
    
    try:
        features = [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
        prediction_idx = int(model.predict(features)[0])
        return {
            "species": SPECIES_MAP.get(prediction_idx, "unknown"),
            "prediction_index": prediction_idx,
            "model_source": "gcs_latest" if os.path.exists(LOCAL_DOWNLOAD_PATH) else "local_fallback"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/train")
async def trigger_training():
    """Triggers the Vertex AI Pipeline to retrain the model."""
    if not os.path.exists(PIPELINE_FILE):
        raise HTTPException(status_code=404, detail=f"Pipeline template not found at {PIPELINE_FILE}")
        
    try:
        logger.info("Submitting PipelineJob to Vertex AI...")
        job = aiplatform.PipelineJob(
            display_name="iris-manual-retrain",
            template_path=PIPELINE_FILE,
            pipeline_root=f"gs://{BUCKET_NAME}/pipeline_root",
            parameter_values={
                "project_id": PROJECT_ID,
                "bucket_name": BUCKET_NAME
            }
        )
        # Using the MLOps runner service account
        job.submit(
            service_account=f"ml-ops-runner@{PROJECT_ID}.iam.gserviceaccount.com"
        )
        return {
            "status": "submitted", 
            "pipeline_job_name": job.name, 
            "dashboard_url": job._dashboard_uri()
        }
    except Exception as e:
        logger.error(f"Failed to submit pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))


        from opentelemetry import trace
from opentelemetry.sdk.resources import RESOURCE_ATTRIBUTES, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor

# --- OPENTELEMETRY SETUP ---
resource = Resource.create(attributes={
    "service.name": "iris-hybrid-service",
    "deployment.environment": "production"
})

provider = TracerProvider(resource=resource)
# This sends traces to your OTel Collector/Tempo
processor = BatchSpanProcessor(OTLPSpanExporter(endpoint="http://otel-collector.observability:4317", insecure=True))
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

# Instrument the App
FastAPIInstrumentor.instrument_app(app)
LoggingInstrumentor().instrument(set_logging_format=True)