from google.cloud import aiplatform

aiplatform.init(project="YOUR_PROJECT_ID", location="us-central1")

job = aiplatform.PipelineJob(
    display_name="iris-training-run",
    template_path="iris_pipeline.yaml",
    pipeline_root="gs://YOUR_BUCKET_NAME/pipeline_root"
)

job.submit()