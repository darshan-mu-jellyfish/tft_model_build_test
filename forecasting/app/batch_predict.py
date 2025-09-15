# import pickle
# from pathlib import Path
# import pandas as pd
# from darts.models import TFTModel
# from app.utils import load_and_preprocess

# def batch_predict(df, model_dir="models/", forecast_horizon=4):
#     series, covariates = load_and_preprocess(df)

#     # Load model + scalers
#     model = TFTModel.load(Path(model_dir) / "tft_model.pth.tar")
#     with open(Path(model_dir) / "scalers.pkl", "rb") as f:
#         scaler_y, scaler_x = pickle.load(f)

#     series_scaled = [scaler_y.transform(s) for s in series]
#     covs_scaled = [scaler_x.transform(c) for c in covariates]

#     forecasts = []
#     for sid, ts, cov in zip(df["series_id_encoded"].unique(), series_scaled, covs_scaled):
#         pred = model.predict(forecast_horizon, past_covariates=cov)
#         pred = scaler_y.inverse_transform(pred)
#         f_df = pred.pd_dataframe().reset_index()
#         f_df["series_id_encoded"] = sid
#         forecasts.append(f_df)

#     return pd.concat(forecasts, ignore_index=True)


# import os
# import pickle
# from pathlib import Path
# import pandas as pd
# from darts.models import TFTModel
# from app.utils import preprocess_series, load_data_from_bq

# def batch_predict(df, forecast_horizon=4):
#     model_dir = os.environ.get("AIP_MODEL_DIR", "models/")
#     output_dir = os.environ.get("AIP_OUTPUT_DIR", "outputs/")
#     Path(output_dir).mkdir(parents=True, exist_ok=True)

#     # Load model + scalers
#     model = TFTModel.load(Path(model_dir) / "tft_model.pth.tar")
#     with open(Path(model_dir) / "scalers.pkl", "rb") as f:
#         scaler_y, scaler_x = pickle.load(f)

#     series, covariates = preprocess_series(df)
#     series_scaled = [scaler_y.transform(s) for s in series]
#     covs_scaled = [scaler_x.transform(c) for c in covariates]

#     forecasts = []
#     for sid, ts, cov in zip(df["series_id_encoded"].unique(), series_scaled, covs_scaled):
#         pred = model.predict(forecast_horizon, past_covariates=cov)
#         pred = scaler_y.inverse_transform(pred)
#         f_df = pred.pd_dataframe().reset_index()
#         f_df["series_id_encoded"] = sid
#         forecasts.append(f_df)

#     results = pd.concat(forecasts, ignore_index=True)

#     output_path = Path(output_dir) / "predictions.csv"
#     results.to_csv(output_path, index=False)
#     print(f"✅ Predictions saved to {output_path}")

# if __name__ == "__main__":
#     PROJECT_ID = os.environ["PROJECT_ID"]
#     DATASET = os.environ["DATASET"]
#     TABLE = os.environ["TABLE"]

#     df = load_data_from_bq(PROJECT_ID, DATASET, TABLE)
#     batch_predict(df)

# app/batch_predict.py
import os
import pickle
from google.cloud import storage
from darts.models import TFTModel

def predict(bucket_name, new_folder, project_id, dataset, table, where=None):
    """Load latest model from GCS and make batch predictions."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Download model and scalers
    model_blob = bucket.blob(f"{new_folder}/tft_model.pth.tar")
    scalers_blob = bucket.blob(f"{new_folder}/scalers.pkl")
    local_model_path = "/tmp/tft_model.pth.tar"
    local_scaler_path = "/tmp/scalers.pkl"
    os.makedirs("/tmp", exist_ok=True)
    model_blob.download_to_filename(local_model_path)
    scalers_blob.download_to_filename(local_scaler_path)

    # Load scalers
    with open(local_scaler_path, "rb") as f:
        scaler_y, scaler_x = pickle.load(f)

    # Load model
    model = TFTModel.load(local_model_path)

    # Load data from BQ
    from app.utils import load_data_from_bq, preprocess_data, scale_series
    df = load_data_from_bq(project_id, dataset, table, where)
    series_list, covariates_list = preprocess_data(df)
    series_scaled, covs_scaled, _, _ = scale_series(series_list, covariates_list)

    # Predict
    forecasts = [model.predict(n=7, past_covariates=cov) for cov in covs_scaled]
    # Reverse scaling
    forecasts = [scaler_y.inverse_transform(f) for f in forecasts]
    return forecasts

