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


import os
import pickle
from pathlib import Path
import pandas as pd
from darts.models import TFTModel
from app.utils import preprocess_series, load_data_from_bq

def batch_predict(df, forecast_horizon=4):
    model_dir = os.environ.get("AIP_MODEL_DIR", "models/")
    output_dir = os.environ.get("AIP_OUTPUT_DIR", "outputs/")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load model + scalers
    model = TFTModel.load(Path(model_dir) / "tft_model.pth.tar")
    with open(Path(model_dir) / "scalers.pkl", "rb") as f:
        scaler_y, scaler_x = pickle.load(f)

    series, covariates = preprocess_series(df)
    series_scaled = [scaler_y.transform(s) for s in series]
    covs_scaled = [scaler_x.transform(c) for c in covariates]

    forecasts = []
    for sid, ts, cov in zip(df["series_id_encoded"].unique(), series_scaled, covs_scaled):
        pred = model.predict(forecast_horizon, past_covariates=cov)
        pred = scaler_y.inverse_transform(pred)
        f_df = pred.pd_dataframe().reset_index()
        f_df["series_id_encoded"] = sid
        forecasts.append(f_df)

    results = pd.concat(forecasts, ignore_index=True)

    output_path = Path(output_dir) / "predictions.csv"
    results.to_csv(output_path, index=False)
    print(f"✅ Predictions saved to {output_path}")

if __name__ == "__main__":
    PROJECT_ID = os.environ["PROJECT_ID"]
    DATASET = os.environ["DATASET"]
    TABLE = os.environ["TABLE"]

    df = load_data_from_bq(PROJECT_ID, DATASET, TABLE)
    batch_predict(df)
