import kfp
from google.cloud import aiplatform
from tft_pipeline import pipeline, PIPELINE_ROOT, PROJECT_ID, REGION

PIPELINE_JSON = "tft_training_pipeline.json"

# Compile pipeline
kfp.compiler.Compiler().compile(
    pipeline_func=pipeline,
    package_path=PIPELINE_JSON
)

# Initialize Vertex AI
aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=PIPELINE_ROOT)

# Submit pipeline run
job = aiplatform.PipelineJob(
    display_name="tft-training-pipeline-job",
    template_path=PIPELINE_JSON,
    pipeline_root=PIPELINE_ROOT,
)
job.run(sync=True)

# Schedule daily
schedule = aiplatform.PipelineJobSchedule.create(
    display_name="tft-training-daily-schedule",
    pipeline_spec_path=PIPELINE_JSON,
    pipeline_root=PIPELINE_ROOT,
    cron="0 6 * * *",
    enable_caching=False
)
print(f"Pipeline scheduled: {schedule.resource_name}")
