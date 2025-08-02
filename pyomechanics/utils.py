from dataclasses import dataclass, Field
import kineticstoolkit.lab as ktk
import networkx as nx
import numpy as np
from typing import Dict, Tuple, List


def generate_marker_graph(marker_names: List[str]):
    g = nx.DiGraph()
    g.add_nodes_from(marker_names)
    g.add_nodes_from([
        "torso_m",
        "thorax_m",
        "shoulder_r",
        "shoulder_l",
        "elbow_r",
        "elbow_l",
        "scapula_r",
        "scapula_l",
        "wrist_r",
        "wrist_l",
        "hip_r",
        "hip_l",
        "pelvis_m",
        "knee_r",
        "knee_l",
        "ankle_r",
        "ankle_l",
        "heel_r",
        "heel_l"
    ], is_custom=True)
    g.add_edges_from([
        ("RSHO", "torso_m"),
        ("LSHO", "torso_m"),
        ("T10", "thorax_m"),
        ("STRN", "thorax_m"),
        ("RSHO", "shoulder_r"),
        ("LSHO", "shoulder_l"),
        ("RELB", "elbow_r"),
        ("RMELB", "elbow_r"),
        ("LELB", "elbow_l"),
        ("LMELB", "elbow_l"),
        ("RSHO", "scapula_r"),
        ("torso_m", "scapula_r"),
        ("LSHO", "scapula_l"),
        ("torso_m", "scapula_l"),
        ("RWRA", "wrist_r"),
        ("RWRB", "wrist_r"),
        ("LWRA", "wrist_l"),
        ("RASI", "hip_r"),
        ("RPSI", "hip_r"),
        ("LASI", "hip_l"),
        ("LPSI", "hip_l"),
        ("hip_l", "pelvis_m"),
        ("hip_r", "pelvis_m"),
        ("RKNE", "knee_r"),
        ("RMKNE", "knee_r"),
        ("LKNE", "knee_l"),
        ("LMKNE", "knee_l"),
        ("RANK", "ankle_r"),
        ("RMANK", "ankle_r"),
        ("LANK", "ankle_l"),
        ("LMANK", "ankle_l"),
        ("RHEE", "heel_r"),
        ("LHEE", "heel_l"),
    ])
    return g


def add_custom_markers(g: nx.DiGraph, markers: ktk.TimeSeries, part_names: List[str]):
    for part_name in part_names:
        if part_name in list(markers.data.keys()):
            marker_name = [name for name in list(markers.data.keys()) if name == part_name][0]
            markers = markers.rename_data(marker_name, part_name)
        else:
            part_name_parents = list(g.predecessors(part_name))
            part_name_parents_avg = np.mean([markers.data[parent] for parent in part_name_parents], axis=0)
            markers = markers.add_data(part_name, part_name_parents_avg, overwrite=True)

    return markers


def subtract_series(a: str, b: str, ts: ktk.TimeSeries):
    return ts.data[a] - ts.data[b]


def parse_c3d_file_path(path: str):
    path_splits = path.split("/")
    user_id = path_splits[-2]
    file_name = path_splits[-1]
    file_name_splits = file_name.split("_")
    user_id = file_name_splits[0]
    session_id = file_name_splits[1]
    height = int(file_name_splits[2])
    weight = int(file_name_splits[3])
    side = file_name_splits[4]
    swing_number = int(file_name_splits[5])
    exit_velo = int(file_name_splits[6].split(".")[0][:2]) + (int(file_name_splits[6].split(".")[0][2]) * .1)
    return user_id, session_id, height, weight, side, swing_number, exit_velo