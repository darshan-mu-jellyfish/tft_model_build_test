import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["train", "predict"])
    parser.add_argument("--project_id", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--table", required=True)
    parser.add_argument("--bucket_name", required=True)
    parser.add_argument("--where", default=None)
    parser.add_argument("--model_folder", default=None)  # for prediction
    parser.add_argument("--output_path", default="predictions.csv")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    os.environ["PROJECT_ID"] = args.project_id
    os.environ["DATASET"] = args.dataset
    os.environ["TABLE"] = args.table
    os.environ["BUCKET_NAME"] = args.bucket_name
    if args.where:
        os.environ["WHERE"] = args.where

    if args.mode == "train":
        from app.train import train_tft_model
        train_tft_model(
            project_id=args.project_id,
            dataset=args.dataset,
            table=args.table,
            bucket_name=args.bucket_name,
            where=args.where,
        )
    elif args.mode == "predict":
        if not args.model_folder:
            raise ValueError("--model_folder is required for prediction")
        from app.batch_predict import predict
        import pandas as pd

        forecasts = predict(
            bucket_name=args.bucket_name,
            model_folder=args.model_folder,
            project_id=args.project_id,
            dataset=args.dataset,
            table=args.table,
            where=args.where,
        )

        all_forecasts = []
        for idx, ts in enumerate(forecasts):
            df_pred = ts.pd_dataframe()
            df_pred["series_idx"] = idx
            all_forecasts.append(df_pred)
        pd.concat(all_forecasts).to_csv(args.output_path, index=False)
