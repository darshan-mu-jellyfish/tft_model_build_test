from kfp import dsl
from kfp.dsl import component

PROJECT_ID = "jfdev-jellyfish-poc"
REGION = "us-central1"
BUCKET = "test_tft-training-bucket-central1"
PIPELINE_ROOT = f"gs://{BUCKET}/tft_pipeline_root/"
IMAGE_URI = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/forecasting/forecasting:latest"

@component(
    base_image="python:3.11",
    packages_to_install=["google-cloud-aiplatform==1.70.0"],
)
def train_component():
    from app.train import train_tft_model
    train_tft_model(
        project_id=PROJECT_ID,
        dataset="test_model_export",
        table="test_table_daily",
        bucket_name=BUCKET
    )

@dsl.pipeline(
    name="tft-training-pipeline",
    pipeline_root=PIPELINE_ROOT
)
def pipeline():
    train_component()
