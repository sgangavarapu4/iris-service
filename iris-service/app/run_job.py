from google.cloud import aiplatform

aiplatform.init(project="project-73119d0a-ddcc-44c1-8f3", location="us-central1")

job = aiplatform.PipelineJob(
    display_name="iris-training-run",
    template_path="app/iris_pipeline.yaml",
    pipeline_root="gs://project-73119d0a-ddcc-44c1-8f3-artifacts/pipeline_root"
)

job.submit()