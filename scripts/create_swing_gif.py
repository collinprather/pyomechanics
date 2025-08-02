import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import kineticstoolkit.lab as ktk
from pathlib import Path
import os
from dotenv import load_dotenv
from pyomechanics.body import joints, parts, shoulder_joint_right, shoulder_joint_left, elbow_joint_right, elbow_joint_left, wrist_joint_right, wrist_joint_left
from pyomechanics.utils import generate_marker_graph, add_custom_markers, parse_c3d_file_path

# Load environment variables
load_dotenv()

def create_swing_gif(session_swing="447_5", output_path="swing_visualization.gif"):
    """
    Create an animated GIF showing 3D bat swing with joint angle plots
    
    Parameters:
    - session_swing: Session swing identifier (e.g., "447_5")
    - output_path: Output file path for the GIF
    """
    
    # Load C3D data
    obp_repo_root_path = os.getenv("obp_repo_root_path")
    c3d_files_path = obp_repo_root_path + "/baseball_hitting/data/c3d"
    c3d_user_sessions = [folder for folder in Path(c3d_files_path).glob("*")]
    c3d_user_session_swings = []
    
    for c3d_user_session in c3d_user_sessions:
        c3d_file_paths = [file for file in c3d_user_session.rglob("*.c3d") if not str(file).endswith("model.c3d")]
        c3d_file_paths_metadata = [(str(path), parse_c3d_file_path(str(path))) for path in c3d_file_paths]
        c3d_file_paths_metadata_sorted = sorted(c3d_file_paths_metadata, key=lambda x: x[1][5])
        c3d_swings = [(f"{int(metadata[1])}_{swing}", path, metadata) for swing, (path, metadata) in enumerate(c3d_file_paths_metadata_sorted, start=1)]
        c3d_user_session_swings.extend(c3d_swings)
    
    # Find the specific session swing
    session_swing_data = [t for t in c3d_user_session_swings if t[0] == session_swing][0]
    session_swing, c3d_file_path, metadata = session_swing_data
    user_id, session_id, height, weight, batter_hand, swing_number, exit_velo = metadata
    
    # Load and process markers
    markers = ktk.read_c3d(c3d_file_path)["Points"]
    markers = ktk.filters.butter(markers, 40, order=4, btype="lowpass")
    
    # Add custom composite markers
    g = generate_marker_graph(list(markers.data.keys()))
    part_names = [node for node, data in g.nodes.data() if data.get("is_custom")]
    markers = add_custom_markers(g, markers, part_names)
    
    # Calculate joint angles
    for part in parts:
        markers.data[part.axis_frames_name] = part.create_axis_frames(markers)
    
    # Calculate joint angles for key joints
    joint_angles = {}
    key_joints = {
        "shoulder_joint_right": shoulder_joint_right,
        "shoulder_joint_left": shoulder_joint_left,
        "elbow_joint_right": elbow_joint_right,
        "elbow_joint_left": elbow_joint_left,
        "wrist_joint_right": wrist_joint_right,
        "wrist_joint_left": wrist_joint_left
    }
    
    for joint_name, joint in key_joints.items():
        angles = joint.angles(markers, batter_hand=batter_hand)
        joint_angles[joint_name] = angles
    
    # Create time array
    time = np.linspace(0, len(markers.data['RSHO']) / 360, len(markers.data['RSHO']))  # Assuming 360 Hz
    
    # Set up the figure with subplots
    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor('white')
    gs = fig.add_gridspec(3, 4, hspace=0.45, wspace=0.3)
    
    # 3D plot takes up left half of the figure
    ax_3d = fig.add_subplot(gs[:, :2], projection='3d')
    
    # Joint angle plots on the right - Lead vs Rear (equal width, using right half)
    ax_shoulder_lead = fig.add_subplot(gs[0, 2:3])
    ax_shoulder_rear = fig.add_subplot(gs[0, 3:4])
    ax_elbow_lead = fig.add_subplot(gs[1, 2:3])
    ax_elbow_rear = fig.add_subplot(gs[1, 3:4])
    ax_wrist_lead = fig.add_subplot(gs[2, 2:3])
    ax_wrist_rear = fig.add_subplot(gs[2, 3:4])
    
    # Define anatomical labels for joint angles
    # Shoulder: YXY sequence = [Plane of elevation, Elevation, Axial rotation]
    # Elbow: ZXY sequence = [Flexion/Extension, Carry angle, Pronation/Supination]  
    # Wrist: ZXY sequence = [Flexion/Extension, Radial/Ulnar deviation, Pronation/Supination]
    
    anatomical_labels = {
        'shoulder': ['Plane of Elevation', 'Elevation', 'Axial Rotation'],
        'elbow': ['Flexion/Extension', 'Carry Angle', 'Pronation/Supination'],
        'wrist': ['Flexion/Extension', 'Radial/Ulnar Dev', 'Pronation/Supination']
    }
    
    # Determine lead vs rear based on batter handedness
    if batter_hand == "R":
        lead_joints = {"shoulder": "shoulder_joint_left", "elbow": "elbow_joint_left", "wrist": "wrist_joint_left"}
        rear_joints = {"shoulder": "shoulder_joint_right", "elbow": "elbow_joint_right", "wrist": "wrist_joint_right"}
        lead_label = "Lead (L)"
        rear_label = "Rear (R)"
    else:
        lead_joints = {"shoulder": "shoulder_joint_right", "elbow": "elbow_joint_right", "wrist": "wrist_joint_right"}
        rear_joints = {"shoulder": "shoulder_joint_left", "elbow": "elbow_joint_left", "wrist": "wrist_joint_left"}
        lead_label = "Lead (R)"
        rear_label = "Rear (L)"
    
    # Define body interconnections for visualization with distinct colors
    interconnections = {
        "UpperArmR": {"Color": [0, 0, 1], "Links": [["RSHO", "elbow_r"]]},  # Blue
        "UpperArmL": {"Color": [0, 0, 1], "Links": [["LSHO", "elbow_l"]]},  # Blue
        "ForearmR": {"Color": [0, 0.8, 0], "Links": [["elbow_r", "wrist_r"]]},  # Green
        "ForearmL": {"Color": [0, 0.8, 0], "Links": [["elbow_l", "wrist_l"]]},  # Green
        "HandR": {"Color": [1, 0.5, 0], "Links": [["wrist_r", "RFIN"]]},  # Orange
        "HandL": {"Color": [1, 0.5, 0], "Links": [["wrist_l", "LFIN"]]},  # Orange
        "Torso": {"Color": [0.5, 0, 0.5], "Links": [["RSHO", "LSHO"], ["RSHO", "hip_r"], ["hip_r", "hip_l"], ["hip_l", "LSHO"]]},  # Purple
        "UpperLegR": {"Color": [1, 0, 0], "Links": [["hip_r", "knee_r"]]},  # Red
        "UpperLegL": {"Color": [1, 0, 0], "Links": [["hip_l", "knee_l"]]},  # Red
        "LowerLegR": {"Color": [1, 1, 0], "Links": [["knee_r", "ankle_r"]]},  # Yellow
        "LowerLegL": {"Color": [1, 1, 0], "Links": [["knee_l", "ankle_l"]]},  # Yellow
        "FootR": {"Color": [0, 1, 1], "Links": [["ankle_r", "heel_r"], ["heel_r", "RTOE"]]},  # Cyan
        "FootL": {"Color": [0, 1, 1], "Links": [["ankle_l", "heel_l"], ["heel_l", "LTOE"]]},  # Cyan
        "bat": {"Color": [0.7, 0.7, 0.7], "Links": [["Marker1", "Marker2"], ["Marker1", "Marker3"], ["Marker2", "Marker3"]]}  # Gray
    }
    
    def animate(frame):
        # Clear all plots
        ax_3d.clear()
        ax_shoulder_lead.clear()
        ax_shoulder_rear.clear()
        ax_elbow_lead.clear()
        ax_elbow_rear.clear()
        ax_wrist_lead.clear()
        ax_wrist_rear.clear()
        
        # 3D visualization
        ax_3d.set_title('3D Baseball Swing Motion', fontsize=14, fontweight='bold')
        ax_3d.set_xlabel('X (m)', fontsize=10)
        ax_3d.set_ylabel('Y (m)', fontsize=10)
        ax_3d.set_zlabel('Z (m)', fontsize=10)
        
        # Plot markers in black with better styling
        for marker_name, marker_data in markers.data.items():
            if marker_name in ["RSHO", "LSHO", "elbow_r", "elbow_l", "wrist_r", "wrist_l", 
                             "hip_r", "hip_l", "knee_r", "knee_l", "ankle_r", "ankle_l",
                             "heel_r", "heel_l", "RTOE", "LTOE", "RFIN", "LFIN",
                             "Marker1", "Marker2", "Marker3"]:
                if not np.isnan(marker_data[frame]).any():
                    ax_3d.scatter(marker_data[frame, 0], marker_data[frame, 1], marker_data[frame, 2], 
                                s=40, alpha=0.9, color='black', edgecolors='white', linewidth=0.5)
        
        # Plot interconnections
        for connection_name, connection_data in interconnections.items():
            color = connection_data["Color"]
            for link in connection_data["Links"]:
                start_marker, end_marker = link
                if start_marker in markers.data and end_marker in markers.data:
                    start_pos = markers.data[start_marker][frame]
                    end_pos = markers.data[end_marker][frame]
                    if not (np.isnan(start_pos).any() or np.isnan(end_pos).any()):
                        ax_3d.plot([start_pos[0], end_pos[0]], 
                                  [start_pos[1], end_pos[1]], 
                                  [start_pos[2], end_pos[2]], 
                                  color=color, linewidth=3, alpha=0.9)
        
        # Set 3D plot limits and view
        ax_3d.set_xlim([-1.5, 1.5])
        ax_3d.set_ylim([-1.5, 1.5])
        ax_3d.set_zlim([0, 2])
        ax_3d.view_init(elev=15, azim=45)
        
        # Add subtle grid and background styling
        ax_3d.grid(True, alpha=0.3)
        ax_3d.xaxis.pane.fill = False
        ax_3d.yaxis.pane.fill = False
        ax_3d.zaxis.pane.fill = False
        ax_3d.xaxis.pane.set_edgecolor('gray')
        ax_3d.yaxis.pane.set_edgecolor('gray')
        ax_3d.zaxis.pane.set_edgecolor('gray')
        ax_3d.xaxis.pane.set_alpha(0.1)
        ax_3d.yaxis.pane.set_alpha(0.1)
        ax_3d.zaxis.pane.set_alpha(0.1)
        
        # Joint angle plots with anatomical labels
        current_time = time[:frame+1]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
        
        # Helper function to plot joint angles
        def plot_joint_angles(ax, joint_name, joint_key, title_suffix, is_lead=True):
            joint_data = joint_angles[joint_key]
            ax.set_title(f'{joint_name} {title_suffix}', fontweight='bold', fontsize=11, pad=10)
            
            # Plot all three angles with anatomical labels
            for i, (color, label) in enumerate(zip(colors, anatomical_labels[joint_name.lower()])):
                if i < joint_data.shape[1]:  # Check if this angle exists
                    ax.plot(current_time, joint_data[:frame+1, i], color=color, 
                           label=label, linewidth=2.5, alpha=0.85)
            
            # Current time indicator
            ax.axvline(time[frame], color='black', linestyle='--', alpha=0.7, linewidth=2)
            
            # Styling
            ax.set_ylabel('Angle (°)', fontsize=10, fontweight='bold')
            ax.legend(loc='lower left', fontsize=8, frameon=True, fancybox=True, shadow=True)
            ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
            ax.tick_params(axis='both', which='major', labelsize=9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(1.5)
            ax.spines['bottom'].set_linewidth(1.5)
        
        # Shoulder angles
        plot_joint_angles(ax_shoulder_lead, 'Shoulder', lead_joints['shoulder'], f'{lead_label}', True)
        plot_joint_angles(ax_shoulder_rear, 'Shoulder', rear_joints['shoulder'], f'{rear_label}', False)
        
        # Elbow angles
        plot_joint_angles(ax_elbow_lead, 'Elbow', lead_joints['elbow'], f'{lead_label}', True)
        plot_joint_angles(ax_elbow_rear, 'Elbow', rear_joints['elbow'], f'{rear_label}', False)
        
        # Wrist angles
        plot_joint_angles(ax_wrist_lead, 'Wrist', lead_joints['wrist'], f'{lead_label}', True)
        plot_joint_angles(ax_wrist_rear, 'Wrist', rear_joints['wrist'], f'{rear_label}', False)
        
        # Add time labels to bottom plots
        ax_wrist_lead.set_xlabel('Time (s)', fontsize=10, fontweight='bold')
        ax_wrist_rear.set_xlabel('Time (s)', fontsize=10, fontweight='bold')
        
        # Add overall title with improved styling
        fig.suptitle(f'Baseball Swing Biomechanics Analysis\nSession {session_swing} ({batter_hand}-Handed Batter)', 
                    fontsize=18, fontweight='bold', y=0.95)
    
    # Create animation
    # Use every 5th frame to speed up animation and reduce file size
    frame_skip = 5
    frames = range(0, len(time), frame_skip)
    
    print(f"Creating animation with {len(frames)} frames...")
    anim = animation.FuncAnimation(fig, animate, frames=frames, interval=150, repeat=True)
    
    # Save as GIF
    print(f"Saving GIF to {output_path}...")
    anim.save(output_path, writer='pillow', fps=8, dpi=80)
    print(f"GIF saved successfully!")
    
    plt.close()

if __name__ == "__main__":
    # Create the visualization
    create_swing_gif(session_swing="447_5", output_path="baseball_swing_analysis.gif")