# import argparse
# import pandas as pd
# from app.train import train_model
# from app.batch_predict import batch_predict
# from app.utils import load_data_from_bq

# def main(mode="train", data_source="bq", project_id=None, dataset=None, table=None, output_path="predictions.csv", where=None):
#     if data_source == "bq":
#         df = load_data_from_bq(project_id, dataset, table, where)
#     else:
#         df = pd.read_csv("train.csv", parse_dates=["timestamp"])  # fallback

#     if mode == "train":
#         train_model(df)
#     elif mode == "predict":
#         preds = batch_predict(df)
#         preds.to_csv(output_path, index=False)
#         print(f"Predictions saved to {output_path}")
#     else:
#         raise ValueError("mode must be 'train' or 'predict'")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--mode", choices=["train", "predict"], required=True)
#     parser.add_argument("--data_source", choices=["bq", "csv"], default="bq")
#     parser.add_argument("--project_id", type=str)
#     parser.add_argument("--dataset", type=str)
#     parser.add_argument("--table", type=str)
#     parser.add_argument("--where", type=str, default=None)
#     parser.add_argument("--output_path", type=str, default="predictions.csv")
#     args = parser.parse_args()

#     main(args.mode, args.data_source, args.project_id, args.dataset, args.table, args.output_path, args.where)


import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=["train", "predict"])
    parser.add_argument("--project_id", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--table", type=str, required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Set env vars (used by train.py & batch_predict.py)
    os.environ["PROJECT_ID"] = args.project_id
    os.environ["DATASET"] = args.dataset
    os.environ["TABLE"] = args.table

    if args.mode == "train":
        from app.train import train_model
        from app.utils import load_data_from_bq
        df = load_data_from_bq(args.project_id, args.dataset, args.table)
        train_model(df)

    elif args.mode == "predict":
        from app.batch_predict import batch_predict
        from app.utils import load_data_from_bq
        df = load_data_from_bq(args.project_id, args.dataset, args.table)
        batch_predict(df)
