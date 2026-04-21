from google.cloud import aiplatform

# 1. Initialize with your specific project details
PROJECT_ID = "project-73119d0a-ddcc-44c1-8f3"
BUCKET_NAME = "project-73119d0a-ddcc-44c1-8f3-artifacts" # Confirm this is your bucket name

aiplatform.init(project=PROJECT_ID, location="us-central1")

job = aiplatform.PipelineJob(
    display_name="iris-training-run",
    template_path="app/iris_pipeline.yaml",
    pipeline_root=f"gs://{BUCKET_NAME}/pipeline_root",
    
    # 2. Pass the mandatory values to the pipeline function here
    parameter_values={
        "project_id": PROJECT_ID,
        "bucket_name": BUCKET_NAME
    }
)

# 3. Submit using your authorized MLOps runner
job.submit(
    service_account=f"ml-ops-runner@{PROJECT_ID}.iam.gserviceaccount.com"
)