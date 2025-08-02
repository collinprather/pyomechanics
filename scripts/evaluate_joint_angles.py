from collections import defaultdict
from dotenv import load_dotenv
import os
import pandas as pd
from sklearn.metrics import root_mean_squared_error, median_absolute_error

load_dotenv()

session_swings_to_skip = {"492_8", "125_4", "492_7", "203_2", "125_6", "125_5", "215_4", "215_5", "203_3", "203_1", "492_6", "203_4", "125_7"}

obp_repo_root_path = os.getenv("obp_repo_root_path")
target_csv_file_path = f"{obp_repo_root_path}/baseball_hitting/data/full_sig/joint_angles.csv"
source_csv_file_path = "./data/output.csv"

df_source = pd.read_csv(source_csv_file_path)
df_target = pd.read_csv(target_csv_file_path)

source_session_swings = set(df_source["session_swing"])
target_session_swings = set(df_target["session_swing"])
session_swings = source_session_swings.intersection(target_session_swings) - session_swings_to_skip

df_target = df_target[df_target["session_swing"].isin(session_swings)]
df_source = df_source[df_source["session_swing"].isin(session_swings)]

df_source["time"] = df_source["time"].round(4)
df_joined = pd.merge(df_source, df_target, how="inner", on=["session_swing", "time"], suffixes=("_source", "_target"))


# first compute the metrics for each joint angle, across all session_swings
metrics = []
for col in [col for col in df_source.columns if col not in ["session_swing", "time"]]:
    metric = {}
    for metric_fn in [root_mean_squared_error, median_absolute_error]:
        metrics.append({
            "col": col,
            "metric": metric_fn.__name__,
            "value": metric_fn(df_joined[f"{col}_target"], df_joined[f"{col}_source"])
        })
pd.DataFrame.from_records(metrics).to_csv(f"./data/results_agged.csv", index=False)
