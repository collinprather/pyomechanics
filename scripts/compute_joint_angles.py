from dotenv import load_dotenv
import kineticstoolkit.lab as ktk
import os
import pandas as pd
from pathlib import Path
from pyomechanics.body import joints, parts
from pyomechanics.utils import generate_marker_graph, add_custom_markers, parse_c3d_file_path


load_dotenv()


obp_repo_root_path = os.getenv("obp_repo_root_path")
c3d_files_path = obp_repo_root_path + "/baseball_hitting/data/c3d"

c3d_user_sessions = [folder for folder in Path(c3d_files_path).glob("*")]
dfs = []
for c3d_user_session in c3d_user_sessions:
    c3d_file_paths = [file for file in c3d_user_session.rglob("*.c3d") if not str(file).endswith("model.c3d")]
    c3d_file_paths_metadata = [(str(path), parse_c3d_file_path(str(path))) for path in c3d_file_paths]
    c3d_file_paths_metadata_sorted = sorted(c3d_file_paths_metadata, key=lambda x: x[1][5]) # sorting by swing number
    c3d_swings = [(f"{int(metadata[1])}_{swing}", path, metadata) for swing, (path, metadata) in enumerate(c3d_file_paths_metadata_sorted, start=1)]
    for session_swing, c3d_file_path, metadata in c3d_swings:
        user_id, session_id, height, weight, batter_hand, swing_number, exit_velo = metadata

        markers = ktk.read_c3d(c3d_file_path)["Points"]
        markers = ktk.filters.butter(markers, 40, order=4, btype="lowpass")

        # adding custom composite markers
        g = generate_marker_graph(list(markers.data.keys()))
        part_names = [node for node, data in g.nodes.data() if data.get("is_custom")]
        markers = add_custom_markers(g, markers, part_names)

        # create new orientation frames (n x 4 x 4) for each body part and add to the timeseries markers
        for part in parts:
            markers = markers.add_data(part.axis_frames_name, part.create_axis_frames(markers), overwrite=True)

        df_joint_angles = pd.DataFrame([(session_swing, t) for t in markers.time], columns=["session_swing", "time"])
        for joint in joints:
            angles = joint.angles(markers)
            for i, var in enumerate(["x", "y", "z"]):
                if not joint.side:
                    # for the pelvis and torso angles, which do not have sides
                    df_joint_angles[f"{joint.__class__.__name__.lower()}_angle_{var}"] = angles[:, i]
                elif batter_hand == joint.side:
                    df_joint_angles[f"rear_{joint.__class__.__name__.lower()}_angle_{var}"] = angles[:, i]
                else:
                    df_joint_angles[f"lead_{joint.__class__.__name__.lower()}_angle_{var}"] = angles[:, i]
        dfs.append(df_joint_angles)

df = pd.concat(dfs, axis=0)    
df.to_csv(f"./data/output.csv", index=False)