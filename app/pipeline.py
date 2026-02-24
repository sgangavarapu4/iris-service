import kfp
from kfp import dsl
from kfp.dsl import component

@component(
    base_image="python:3.9", 
    packages_to_install=["scikit-learn", "joblib", "google-cloud-storage"]
)
def train_iris_task(project_id: str, bucket_name: str):
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    from google.cloud import storage
    import os
    
    # 1. Train
    data = load_iris()
    model = RandomForestClassifier()
    model.fit(data.data, data.target)
    
    # 2. Save locally in the temporary container
    local_file = "model.pkl"
    joblib.dump(model, local_file)
    
    # 3. Upload to GCS
    # This uses the default credentials assigned to the Vertex AI worker
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob("models/iris_model_v1.pkl")
    blob.upload_from_filename(local_file)
    
    print(f"Model successfully uploaded to gs://{bucket_name}/models/iris_model_v1.pkl")

@dsl.pipeline(name="iris-retrain")
def iris_pipeline(project_id: str, bucket_name: str):
    # Pass the parameters down to the task
    train_iris_task(project_id=project_id, bucket_name=bucket_name)

if __name__ == "__main__":
    kfp.compiler.Compiler().compile(iris_pipeline, "iris_pipeline.yaml")