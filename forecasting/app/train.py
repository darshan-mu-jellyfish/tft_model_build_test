import os
import pickle
from datetime import datetime
from darts.models import TFTModel

def train_tft_model(project_id, dataset, table, bucket_name, where=None, model_dir=None):
    """
    Train the TFT model and save to model_dir (GCS path supported)
    """
    import pandas as pd
    from google.cloud import storage
    from app.utils import load_data_from_bq, preprocess_data, scale_series

    # Load and preprocess data
    df = load_data_from_bq(project_id, dataset, table, where)
    series_list, covariates_list = preprocess_data(df)
    series_scaled, covs_scaled, scaler_y, scaler_x = scale_series(series_list, covariates_list)

    # Train model
    model = TFTModel(
        input_chunk_length=30,
        output_chunk_length=7,
        hidden_size=64,
        lstm_layers=1,
        batch_size=32,
        n_epochs=5,  # adjust for testing
    )
    model.fit(series_scaled, past_covariates=covs_scaled)

    # Save locally
    local_model_path = "/tmp/tft_model.pth.tar"
    local_scaler_path = "/tmp/scalers.pkl"
    model.save(local_model_path)
    with open(local_scaler_path, "wb") as f:
        pickle.dump((scaler_y, scaler_x), f)

    # Upload to GCS
    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # versioned folder
    if model_dir is None:
        version_folder = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        model_dir = f"darts_models/tft_model/{version_folder}"

    bucket.blob(f"{model_dir}/tft_model.pth.tar").upload_from_filename(local_model_path)
    bucket.blob(f"{model_dir}/scalers.pkl").upload_from_filename(local_scaler_path)
