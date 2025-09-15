import os
import pickle
from google.cloud import storage
from darts.models import TFTModel

def predict(bucket_name, model_folder, project_id, dataset, table, where=None):
    """Load specified model from GCS and make batch predictions"""
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Download model + scalers
    model_blob = bucket.blob(f"{model_folder}/tft_model.pth.tar")
    scalers_blob = bucket.blob(f"{model_folder}/scalers.pkl")
    os.makedirs("/tmp", exist_ok=True)
    model_blob.download_to_filename("/tmp/tft_model.pth.tar")
    scalers_blob.download_to_filename("/tmp/scalers.pkl")

    # Load scalers
    with open("/tmp/scalers.pkl", "rb") as f:
        scaler_y, scaler_x = pickle.load(f)

    # Load model
    model = TFTModel.load("/tmp/tft_model.pth.tar")

    from app.utils import load_data_from_bq, preprocess_data, scale_series
    df = load_data_from_bq(project_id, dataset, table, where)
    series_list, covariates_list = preprocess_data(df)
    series_scaled, covs_scaled, _, _ = scale_series(series_list, covariates_list)

    # Predict
    forecasts = [model.predict(n=7, past_covariates=cov) for cov in covs_scaled]
    forecasts = [scaler_y.inverse_transform(f) for f in forecasts]
    return forecasts
