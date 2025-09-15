# import pickle
# from pathlib import Path
# from darts.models import TFTModel
# from app.utils import load_and_preprocess, scale_series

# def train_model(df, model_dir="models/", forecast_horizon=4):
#     series, covariates = load_and_preprocess(df)
#     series_scaled, covs_scaled, scaler_y, scaler_x = scale_series(series, covariates)

#     # Train single global TFT across series
#     model = TFTModel(
#         input_chunk_length=24,
#         output_chunk_length=forecast_horizon,
#         hidden_size=16,
#         n_epochs=10,
#         random_state=42,
#         add_relative_index=True
#     )

#     model.fit(series_scaled, past_covariates=covs_scaled, verbose=True)

#     # Save model + scalers
#     Path(model_dir).mkdir(parents=True, exist_ok=True)
    
#     model.save(str(Path(model_dir) / "tft_model.pth.tar"))
    
#     with open(Path(model_dir) / "scalers.pkl", "wb") as f:
#         pickle.dump((scaler_y, scaler_x), f)

#     return model

# import os
# import pickle
# from pathlib import Path
# from darts.models import TFTModel
# from app.utils import preprocess_series, scale_series, load_data_from_bq

# def train_model(df, forecast_horizon=4):
#     # Preprocess
#     series, covariates = preprocess_series(df)
#     series_scaled, covs_scaled, scaler_y, scaler_x = scale_series(series, covariates)

#     # Train TFT
#     model = TFTModel(
#         input_chunk_length=24,
#         output_chunk_length=forecast_horizon,
#         hidden_size=16,
#         n_epochs=10,
#         random_state=42,
#         add_relative_index=True
#     )
#     model.fit(series_scaled, past_covariates=covs_scaled, verbose=True)

#     # Vertex AI model dir
#     model_dir = os.environ.get("AIP_MODEL_DIR", "models/")
#     Path(model_dir).mkdir(parents=True, exist_ok=True)

#     # Save model + scalers
#     model.save(Path(model_dir) / "tft_model.pth.tar")
#     with open(Path(model_dir) / "scalers.pkl", "wb") as f:
#         pickle.dump((scaler_y, scaler_x), f)

#     print(f"✅ Model artifacts saved to {model_dir}")

# if __name__ == "__main__":
#     PROJECT_ID = os.environ["PROJECT_ID"]
#     DATASET = os.environ["DATASET"]
#     TABLE = os.environ["TABLE"]

#     df = load_data_from_bq(PROJECT_ID, DATASET, TABLE)
#     train_model(df)


# app/train.py
import os
import pickle
from datetime import datetime
from google.cloud import storage
import torch
from darts.models import TFTModel
from app.utils import load_data_from_bq, preprocess_data, scale_series

# ------------------------
# PyTorch 2.6 patch
# ------------------------
_real_torch_load = torch.load
def torch_load_patch(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _real_torch_load(*args, **kwargs)
torch.load = torch_load_patch

def train_model(df):
    # Preprocess
    series_list, covariates_list = preprocess_data(df)
    series_scaled, covs_scaled, scaler_y, scaler_x = scale_series(series_list, covariates_list)

    # Train TFT Model
    model = TFTModel(
        input_chunk_length=24,
        output_chunk_length=7,
        hidden_size=16,
        n_epochs=10,
        random_state=42,
        add_relative_index=True
    )
    model.fit(series_scaled, past_covariates=covs_scaled, verbose=True)

    # Local temporary paths
    os.makedirs("/tmp/model", exist_ok=True)
    local_model_path = "/tmp/model/tft_model.pth.tar"
    local_scaler_path = "/tmp/model/scalers.pkl"
    model.save(local_model_path)
    with open(local_scaler_path, "wb") as f:
        pickle.dump([scaler_y, scaler_x], f)

    # GCS Versioning
    BUCKET_NAME = os.environ["BUCKET_NAME"]
    MODEL_BASE = "darts_models/tft_model"
    OLD_FOLDER = f"{MODEL_BASE}/Old_Models"
    NEW_FOLDER = f"{MODEL_BASE}/New_Models"
    ARCHIVE_FOLDER = f"{MODEL_BASE}/Archive"

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)

    # Move previous new models to old and archive
    blobs_new = list(bucket.list_blobs(prefix=NEW_FOLDER))
    if blobs_new:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for blob in blobs_new:
            old_name = blob.name.replace(NEW_FOLDER, OLD_FOLDER)
            bucket.rename_blob(blob, old_name)
            archive_name = blob.name.replace(NEW_FOLDER, f"{ARCHIVE_FOLDER}/tft_model_{timestamp}")
            bucket.copy_blob(bucket.blob(old_name), bucket, new_name=archive_name)

    # Upload new model & scalers
    bucket.blob(f"{NEW_FOLDER}/tft_model.pth.tar").upload_from_filename(local_model_path)
    bucket.blob(f"{NEW_FOLDER}/scalers.pkl").upload_from_filename(local_scaler_path)

    print(f"✅ New model uploaded to gs://{BUCKET_NAME}/{NEW_FOLDER}")
    print(f"✅ Old models stored in gs://{BUCKET_NAME}/{OLD_FOLDER} and archived in gs://{BUCKET_NAME}/{ARCHIVE_FOLDER}/")

def main():
    PROJECT_ID = os.environ["PROJECT_ID"]
    DATASET = os.environ["DATASET"]
    TABLE = os.environ["TABLE"]
    WHERE = os.environ.get("WHERE")
    df = load_data_from_bq(PROJECT_ID, DATASET, TABLE, where=WHERE)
    train_model(df)

if __name__ == "__main__":
    main()
