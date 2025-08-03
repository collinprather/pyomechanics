import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import kineticstoolkit.lab as ktk
from pathlib import Path
import os
import sys
import argparse
from dotenv import load_dotenv
from pyomechanics.body import joints, parts, shoulder_joint_right, shoulder_joint_left, elbow_joint_right, elbow_joint_left, wrist_joint_right, wrist_joint_left
from pyomechanics.utils import generate_marker_graph, add_custom_markers, parse_c3d_file_path

# Set font to Courier New like plot_mae.py
plt.rcParams['font.family'] = 'Courier New'

# Load environment variables
load_dotenv()

def create_swing_gif(session_swing="447_5", output_path=None):
    if output_path is None:
        output_path = f"viz/swing_and_joint_angles_{session_swing}.gif"
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
    print(f"Session Swing: {session_swing}, User ID: {user_id}, Session ID: {session_id}, Height: {height}, Weight: {weight}, Batter Hand: {batter_hand}, Swing Number: {swing_number}, Exit Velo: {exit_velo}")
    
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
    
    # Load ground truth joint angles from Driveline data
    ground_truth_angles = {}
    try:
        target_csv_file_path = f"{obp_repo_root_path}/baseball_hitting/data/full_sig/joint_angles.csv"
        df_target = pd.read_csv(target_csv_file_path)
        df_target = df_target[df_target["session_swing"] == session_swing]
        
        if not df_target.empty:
            # Map Driveline naming to our joint naming
            driveline_to_our_mapping = {
                'lead_shoulder': 'shoulder_joint_left' if batter_hand == 'R' else 'shoulder_joint_right',
                'rear_shoulder': 'shoulder_joint_right' if batter_hand == 'R' else 'shoulder_joint_left',
                'lead_elbow': 'elbow_joint_left' if batter_hand == 'R' else 'elbow_joint_right',
                'rear_elbow': 'elbow_joint_right' if batter_hand == 'R' else 'elbow_joint_left',
                'lead_wrist': 'wrist_joint_left' if batter_hand == 'R' else 'wrist_joint_right',
                'rear_wrist': 'wrist_joint_right' if batter_hand == 'R' else 'wrist_joint_left'
            }
            
            for driveline_name, our_name in driveline_to_our_mapping.items():
                # Extract x, y, z angles for this joint
                x_angles = df_target[f"{driveline_name}_angle_x"].values
                y_angles = df_target[f"{driveline_name}_angle_y"].values
                z_angles = df_target[f"{driveline_name}_angle_z"].values
                
                # Stack into same format as our calculated angles
                ground_truth_angles[our_name] = np.column_stack([x_angles, y_angles, z_angles])
            
            print(f"Loaded ground truth data for session {session_swing}")
        else:
            print(f"No ground truth data found for session {session_swing}")
    except Exception as e:
        print(f"Could not load ground truth data: {e}")
        ground_truth_angles = {}
    
    # Create time array
    time = np.linspace(0, len(markers.data['RSHO']) / 360, len(markers.data['RSHO']))  # Assuming 360 Hz
    
    # Calculate fixed x-axis limits for consistent plotting
    start_frame = 250
    x_min = time[start_frame]
    x_max = time[-1]
    
    # Calculate fixed y-axis limits for consistent joint angle plotting
    y_limits = {}
    for joint_name, joint_data in joint_angles.items():
        # Get min and max across all angles for this joint
        joint_min = np.min(joint_data[start_frame:])
        joint_max = np.max(joint_data[start_frame:])
        
        # Include ground truth data in limits calculation if available
        if joint_name in ground_truth_angles:
            gt_data = ground_truth_angles[joint_name]
            joint_min = min(joint_min, np.min(gt_data[start_frame:]))
            joint_max = max(joint_max, np.max(gt_data[start_frame:]))
        
        # Add some padding
        padding = (joint_max - joint_min) * 0.1
        y_limits[joint_name] = [joint_min - padding, joint_max + padding]
    
    # Set up the figure with subplots - black background like body frame viz
    fig = plt.figure(figsize=(20, 12), facecolor='black')
    gs = fig.add_gridspec(3, 4, hspace=0.45, wspace=0.3)
    
    # 3D plot takes up left half of the figure - black background
    ax_3d = fig.add_subplot(gs[:, :2], projection='3d', facecolor='black')
    
    # Joint angle plots on the right - Lead vs Rear (equal width, using right half) - black backgrounds
    ax_shoulder_lead = fig.add_subplot(gs[0, 2:3], facecolor='black')
    ax_shoulder_rear = fig.add_subplot(gs[0, 3:4], facecolor='black')
    ax_elbow_lead = fig.add_subplot(gs[1, 2:3], facecolor='black')
    ax_elbow_rear = fig.add_subplot(gs[1, 3:4], facecolor='black')
    ax_wrist_lead = fig.add_subplot(gs[2, 2:3], facecolor='black')
    ax_wrist_rear = fig.add_subplot(gs[2, 3:4], facecolor='black')
    
    # Define anatomical labels for joint angles
    # Shoulder: YXY sequence = [Plane of elevation, Elevation, Axial rotation]
    # Elbow: ZXY sequence = [Flexion/Extension, Carry angle, Pronation/Supination]  
    # Wrist: ZXY sequence = [Flexion/Extension, Radial/Ulnar deviation, Pronation/Supination]
    
    anatomical_labels = {
        'shoulder': ['Horizontal Ab (+)/Adduction (-)', 'Ab (+)/Adduction (-)', 'External (+)/Internal (-) Rotation'],
        'elbow': ['Flexion (+)/Extension (-)', 'Constrained', 'Pronation (+)/Supination (-)'],
        'wrist': ['	Flexion (-)/Extension (+)', 'Ulnar (-)/Radial (+) Deviation', 'Constrained']
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
        # Clear all plots and set black backgrounds
        ax_3d.clear()
        ax_3d.set_facecolor('black')
        ax_shoulder_lead.clear()
        ax_shoulder_lead.set_facecolor('black')
        ax_shoulder_rear.clear()
        ax_shoulder_rear.set_facecolor('black')
        ax_elbow_lead.clear()
        ax_elbow_lead.set_facecolor('black')
        ax_elbow_rear.clear()
        ax_elbow_rear.set_facecolor('black')
        ax_wrist_lead.clear()
        ax_wrist_lead.set_facecolor('black')
        ax_wrist_rear.clear()
        ax_wrist_rear.set_facecolor('black')
        
        # 3D visualization with white text on black background
        ax_3d.set_title('3D Baseball Swing Motion', fontsize=14, fontweight='bold', color='white')
        ax_3d.set_xlabel('X (m)', fontsize=10, color='white')
        ax_3d.set_ylabel('Y (m)', fontsize=10, color='white')
        ax_3d.set_zlabel('Z (m)', fontsize=10, color='white')
        
        # Plot markers in white on black background - including head markers
        for marker_name, marker_data in markers.data.items():
            if marker_name in ["RSHO", "LSHO", "elbow_r", "elbow_l", "wrist_r", "wrist_l", 
                             "hip_r", "hip_l", "knee_r", "knee_l", "ankle_r", "ankle_l",
                             "heel_r", "heel_l", "RTOE", "LTOE", "RFIN", "LFIN",
                             "RFHD", "LFHD", "RBHD", "LBHD", "torso_m",
                             "Marker1", "Marker2", "Marker3"]:
                if not np.isnan(marker_data[frame]).any():
                    ax_3d.scatter(marker_data[frame, 0], marker_data[frame, 1], marker_data[frame, 2], 
                                s=25, alpha=0.75, color='white')
        
        # Plot body frame connections - using blue for body, white for bat like body frame viz
        # Include head connections like visualize_body_frame.py
        body_connections = [
            # Torso
            ('RSHO', 'LSHO'), ('RSHO', 'torso_m'), ('LSHO', 'torso_m'),
            ('torso_m', 'hip_r'), ('torso_m', 'hip_l'), ('hip_r', 'hip_l'),
            # Hip to shoulder connections
            ('hip_r', 'RSHO'), ('hip_l', 'LSHO'),
            # Arms
            ('RSHO', 'elbow_r'), ('elbow_r', 'wrist_r'), ('wrist_r', 'RFIN'),
            ('LSHO', 'elbow_l'), ('elbow_l', 'wrist_l'), ('wrist_l', 'LFIN'),
            # Legs
            ('hip_r', 'knee_r'), ('knee_r', 'ankle_r'), ('ankle_r', 'heel_r'), ('heel_r', 'RTOE'),
            ('hip_l', 'knee_l'), ('knee_l', 'ankle_l'), ('ankle_l', 'heel_l'), ('heel_l', 'LTOE'),
            # Head
            ('RFHD', 'LFHD'), ('RFHD', 'RBHD'), ('LFHD', 'LBHD'), ('RBHD', 'LBHD'),
            # Head to torso
            ('RFHD', 'torso_m'), ('LFHD', 'torso_m'), ('RBHD', 'torso_m'), ('LBHD', 'torso_m')
        ]
        bat_connections = [('Marker1', 'Marker2'), ('Marker1', 'Marker3'), ('Marker2', 'Marker3')]
        
        # Draw body connections in blue
        for marker1, marker2 in body_connections:
            if marker1 in markers.data and marker2 in markers.data:
                data1 = markers.data[marker1]
                data2 = markers.data[marker2]
                if not np.isnan(data1[frame]).any() and not np.isnan(data2[frame]).any():
                    ax_3d.plot([data1[frame, 0], data2[frame, 0]], 
                               [data1[frame, 1], data2[frame, 1]], 
                               [data1[frame, 2], data2[frame, 2]], 
                               color='tab:blue', alpha=0.9, linewidth=3)
        
        # Draw bat connections in white
        for marker1, marker2 in bat_connections:
            if marker1 in markers.data and marker2 in markers.data:
                data1 = markers.data[marker1]
                data2 = markers.data[marker2]
                if not np.isnan(data1[frame]).any() and not np.isnan(data2[frame]).any():
                    ax_3d.plot([data1[frame, 0], data2[frame, 0]], 
                               [data1[frame, 1], data2[frame, 1]], 
                               [data1[frame, 2], data2[frame, 2]], 
                               color='white', alpha=0.75, linewidth=3)
        
        # Set 3D plot limits and view
        ax_3d.set_xlim([-1.5, 1.5])
        ax_3d.set_ylim([-1.5, 1.5])
        ax_3d.set_zlim([0, 2])
        ax_3d.view_init(elev=15, azim=35 if batter_hand == "L" else -35)
        
        # Set tick colors to white and add black panes with light grid like body frame viz
        ax_3d.tick_params(colors='white')
        ax_3d.xaxis.pane.fill = False
        ax_3d.yaxis.pane.fill = False
        ax_3d.zaxis.pane.fill = False
        ax_3d.xaxis.pane.set_edgecolor('black')
        ax_3d.yaxis.pane.set_edgecolor('black')
        ax_3d.zaxis.pane.set_edgecolor('black')
        ax_3d.grid(True, color='lightgrey', alpha=0.15)
        
        # Joint angle plots with anatomical labels - using red/green/blue like joint angle viz
        current_time = time[:frame+1]
        colors = ['red', 'green', 'blue']  # Red, Green, Blue like joint angle viz
        
        # Helper function to plot joint angles with black background styling
        def plot_joint_angles(ax, joint_name, joint_key, title_suffix, is_lead=True):
            joint_data = joint_angles[joint_key]
            ax.set_title(f'{joint_name} {title_suffix}', fontweight='bold', fontsize=11, pad=10, color='white')
            
            # Set fixed x-axis and y-axis limits for all frames
            ax.set_xlim([x_min, x_max])
            ax.set_ylim(y_limits[joint_key])
            
            # Plot all three angles with anatomical labels (solid lines for our estimates)
            for i, (color, label) in enumerate(zip(colors, anatomical_labels[joint_name.lower()])):
                if i < joint_data.shape[1]:  # Check if this angle exists
                    ax.plot(current_time, joint_data[:frame+1, i], color=color, 
                           label=label, linewidth=3, alpha=0.9, linestyle='-')
            
            # Plot ground truth angles if available (dashed lines)
            if joint_key in ground_truth_angles:
                gt_data = ground_truth_angles[joint_key]
                for i, color in enumerate(colors):
                    if i < gt_data.shape[1] and frame+1 <= len(gt_data):
                        ax.plot(current_time, gt_data[:frame+1, i], color=color, 
                               linewidth=2, alpha=0.7, linestyle='--')
            
            # Current time indicator in white
            ax.axvline(time[frame], color='white', linestyle='--', alpha=0.7, linewidth=2)
            
            # Styling with white text and spines
            ax.set_ylabel('Angle (°)', fontsize=10, fontweight='bold', color='white')
            ax.legend(loc='lower left', fontsize=8, frameon=True, facecolor='black', 
                     edgecolor='white', labelcolor='white')
            ax.grid(True, color='lightgrey', alpha=0.3)
            ax.tick_params(colors='white', labelsize=9)
            for spine in ax.spines.values():
                spine.set_color('white')
            # Add horizontal line at zero
            ax.axhline(y=0, color='white', linestyle='-', alpha=0.5, linewidth=1)
        
        # Shoulder angles
        plot_joint_angles(ax_shoulder_lead, 'Shoulder', lead_joints['shoulder'], f'{lead_label}', True)
        plot_joint_angles(ax_shoulder_rear, 'Shoulder', rear_joints['shoulder'], f'{rear_label}', False)
        
        # Elbow angles
        plot_joint_angles(ax_elbow_lead, 'Elbow', lead_joints['elbow'], f'{lead_label}', True)
        plot_joint_angles(ax_elbow_rear, 'Elbow', rear_joints['elbow'], f'{rear_label}', False)
        
        # Wrist angles
        plot_joint_angles(ax_wrist_lead, 'Wrist', lead_joints['wrist'], f'{lead_label}', True)
        plot_joint_angles(ax_wrist_rear, 'Wrist', rear_joints['wrist'], f'{rear_label}', False)
        
        # Add time labels to bottom plots with white color
        ax_wrist_lead.set_xlabel('Time (s)', fontsize=10, fontweight='bold', color='white')
        ax_wrist_rear.set_xlabel('Time (s)', fontsize=10, fontweight='bold', color='white')
        
        # Add legend for line styles in top right corner
        legend_elements = [
            plt.Line2D([0], [0], color='white', linewidth=2, linestyle='-', label='My Estimates'),
            plt.Line2D([0], [0], color='white', linewidth=2, linestyle='--', label='Driveline Ground Truth')
        ]
        fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98), 
                  fontsize=12, frameon=True, facecolor='black', edgecolor='white', 
                  labelcolor='white')
        
        # Add overall title with white color on black background
        title_line1 = f'Baseball Swing Biomechanics Analysis - Session {session_swing}'
        title_line2 = f'{batter_hand}-Handed | Height: {height}" | Weight: {weight}lbs | Exit Velo: {exit_velo}mph'
        fig.suptitle(f'{title_line1}\n{title_line2}', 
                    fontsize=16, fontweight='bold', y=0.95, color='white')
    
    # Create animation
    # Use every 5th frame to speed up animation and reduce file size
    frame_skip = 10
    base_frames = list(range(len(time) - (100 * frame_skip), len(time), frame_skip))
    
    # Add pause at the end by duplicating the final frame multiple times
    # Add about 1 second pause (10 extra frames at 10 fps = 1 second)
    final_frame = base_frames[-1]
    pause_frames = [final_frame] * 10
    frames = base_frames + pause_frames
    
    print(f"Creating animation with {len(frames)} frames...")
    anim = animation.FuncAnimation(fig, animate, frames=frames, interval=150, repeat=True)
    
    # Save as GIF
    print(f"Saving GIF to {output_path}...")
    anim.save(output_path, writer='pillow', fps=10, dpi=80)
    print(f"GIF saved successfully!")
    
    plt.close()

if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Create animated GIF of baseball swing with joint angles')
    parser.add_argument('session_swing', help='Session swing identifier (e.g., "215_2")')
    parser.add_argument('--output', '-o', help='Output path for the GIF (optional)')
    
    args = parser.parse_args()
    
    # Create the visualization
    create_swing_gif(session_swing=args.session_swing, output_path=args.output)