from dotenv import load_dotenv
import matplotlib.pyplot as plt
import os
import pandas as pd

load_dotenv()


obp_repo_root_path = os.getenv("obp_repo_root_path")
target_csv_file_path = f"{obp_repo_root_path}/baseball_hitting/data/full_sig/joint_angles.csv"
source_csv_file_path = "./data/output.csv"
df_source = pd.read_csv(source_csv_file_path)
df_target = pd.read_csv(target_csv_file_path)
session_swing = "215_2"
df_source = df_source[df_source["session_swing"] == session_swing]
df_target = df_target[df_target["session_swing"] == session_swing]

joints = [
    "shoulder",
    "elbow", 
    "wrist", 
    "hip", 
    "knee", 
    "ankle"
]
nrows = len(joints)
ncols = 2
fig, ax = plt.subplots(nrows, ncols, figsize=(10, nrows*3.5))

for i, joint in enumerate(joints):
    for var, color in [("x", "tab:blue"), ("y", "tab:green"), ("z", "tab:purple")]:
        ax[i][0].set_title(f"Lead {joint} angles")
        ax[i][0].axhline(0, c="grey", alpha=0.5, ls="dotted")
        ax[i][0].plot(df_source["time"], df_source[f"lead_{joint}_angle_{var}"], c=color, alpha=0.8)
        ax[i][0].plot(df_target["time"], df_target[f"lead_{joint}_angle_{var}"], c=color, alpha=0.8, ls="dashed")

        ax[i][1].set_title(f"Rear {joint} angles")
        ax[i][1].axhline(0, c="grey", alpha=0.5, ls="dotted")
        ax[i][1].plot(df_source["time"], df_source[f"rear_{joint}_angle_{var}"], c=color, alpha=0.8)
        ax[i][1].plot(df_target["time"], df_target[f"rear_{joint}_angle_{var}"], c=color, alpha=0.8, ls="dashed")

# will need to implement custom legend
fig.suptitle(f"{session_swing} joint angles")
plt.show()
