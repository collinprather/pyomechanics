import kineticstoolkit.lab as ktk
import os
from dotenv import load_dotenv
from pathlib import Path
from pyomechanics.utils import generate_marker_graph, add_custom_markers, parse_c3d_file_path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def visualize_body_frame(session_swing="447_5", output_path="markers_visualization.gif"):
    """
    Visualize 3D body marker points using matplotlib and save as GIF
    
    Parameters:
    - session_swing: Session swing identifier (e.g., "447_5")
    - output_path: Output file path for the GIF
    """
    
    # Load environment variables
    load_dotenv()
    
    # Load C3D data using the same logic as compute_joint_angles.py
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
    
    print(f"Loading C3D file: {c3d_file_path}")
    
    # Load and process markers
    markers = ktk.read_c3d(c3d_file_path)["Points"]
    markers = ktk.filters.butter(markers, 40, order=4, btype="lowpass")
    
    # Add custom composite markers
    g = generate_marker_graph(list(markers.data.keys()))
    part_names = [node for node, data in g.nodes.data() if data.get("is_custom")]
    markers = add_custom_markers(g, markers, part_names)
    
    print(f"Loaded {len(markers.data)} markers with {len(markers.time)} time points")
    
    # Define body frame markers for the plot
    body_frame_markers = ['RSHO', 'LSHO', 'elbow_r', 'elbow_l', 'wrist_r', 'wrist_l', 
                          'hip_r', 'hip_l', 'knee_r', 'knee_l', 'ankle_r', 'ankle_l',
                          'heel_r', 'heel_l', 'LTOE', 'RTOE', 'torso_m', 'RFHD', 'LFHD', 'RBHD', 'LBHD']
    
    # Filter to only markers that exist in the data
    available_body_markers = [m for m in body_frame_markers if m in markers.data]
    
    # Use all available markers for axis bounds calculation (same as single script)
    available_markers = list(markers.data.keys())
    
    # Calculate global bounds across all markers and all time points
    all_positions = []
    for marker_name in available_markers:
        marker_data = markers.data[marker_name]
        # Get all non-NaN positions across all time points
        valid_mask = ~np.isnan(marker_data).any(axis=1)
        if valid_mask.any():
            all_positions.append(marker_data[valid_mask])
    
    if all_positions:
        all_positions = np.vstack(all_positions)
        margin = 0.1  # Small margin around markers
        
        # Calculate ranges for each axis
        x_range = all_positions[:, 0].max() - all_positions[:, 0].min()
        y_range = all_positions[:, 1].max() - all_positions[:, 1].min()
        z_range = all_positions[:, 2].max() - all_positions[:, 2].min()
        
        # Use the maximum range to make all axes equivalent
        max_range = max(x_range, y_range, z_range)
        
        # Calculate centers
        x_center = (all_positions[:, 0].min() + all_positions[:, 0].max()) / 2
        y_center = (all_positions[:, 1].min() + all_positions[:, 1].max()) / 2
        z_center = (all_positions[:, 2].min() + all_positions[:, 2].max()) / 2
        
        # Set equal ranges around centers
        half_range = max_range / 2 + margin
        x_min, x_max = x_center - half_range, x_center + half_range
        y_min, y_max = y_center - half_range, y_center + half_range
        z_min, z_max = z_center - half_range, z_center + half_range
    else:
        # Fallback to original limits if no valid positions found
        x_min, x_max = -1.5, 1.5
        y_min, y_max = -1.5, 1.5
        z_min, z_max = 0, 2
    
    # Define body frame connections (lines between markers)
    body_connections = [
        # Torso
        ('RSHO', 'LSHO'), ('RSHO', 'torso_m'), ('LSHO', 'torso_m'),
        ('torso_m', 'hip_r'), ('torso_m', 'hip_l'), ('hip_r', 'hip_l'),
        # Hip to shoulder connections
        ('hip_r', 'RSHO'), ('hip_l', 'LSHO'),
        # Arms
        ('RSHO', 'elbow_r'), ('elbow_r', 'wrist_r'),
        ('LSHO', 'elbow_l'), ('elbow_l', 'wrist_l'),
        # Legs
        ('hip_r', 'knee_r'), ('knee_r', 'ankle_r'), ('ankle_r', 'heel_r'), ('heel_r', 'RTOE'),
        ('hip_l', 'knee_l'), ('knee_l', 'ankle_l'), ('ankle_l', 'heel_l'), ('heel_l', 'LTOE'),
        # Head
        ('RFHD', 'LFHD'), ('RFHD', 'RBHD'), ('LFHD', 'LBHD'), ('RBHD', 'LBHD'),
        # Head to torso
        ('RFHD', 'torso_m'), ('LFHD', 'torso_m'), ('RBHD', 'torso_m'), ('LBHD', 'torso_m')
    ]
    bat_connections = [('Marker1', 'Marker2'), ('Marker1', 'Marker3'), ('Marker2', 'Marker3')]
    
    # Set up the figure with black background
    fig = plt.figure(figsize=(12, 9), facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')
    
    # Minimize margins around the plot
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    
    def animate(frame):
        ax.clear()
        ax.set_facecolor('black')
        
        # Plot body frame markers
        for marker_name in available_body_markers:
            marker_data = markers.data[marker_name]
            if not np.isnan(marker_data[frame]).any():
                ax.scatter(marker_data[frame, 0], marker_data[frame, 1], marker_data[frame, 2], 
                          s=25, alpha=0.75, color='white')
        
        # Draw connections between body frame markers
        for marker1, marker2 in body_connections:
            if marker1 in markers.data and marker2 in markers.data:
                data1 = markers.data[marker1]
                data2 = markers.data[marker2]
                if not np.isnan(data1[frame]).any() and not np.isnan(data2[frame]).any():
                    ax.plot([data1[frame, 0], data2[frame, 0]], 
                           [data1[frame, 1], data2[frame, 1]], 
                           [data1[frame, 2], data2[frame, 2]], 
                           color='tab:blue', alpha=0.9, linewidth=3)
        for marker1, marker2 in bat_connections:
            if marker1 in markers.data and marker2 in markers.data:
                data1 = markers.data[marker1]
                data2 = markers.data[marker2]
                if not np.isnan(data1[frame]).any() and not np.isnan(data2[frame]).any():
                    ax.plot([data1[frame, 0], data2[frame, 0]], 
                           [data1[frame, 1], data2[frame, 1]], 
                           [data1[frame, 2], data2[frame, 2]], 
                           color='white', alpha=0.75, linewidth=3)
        
        # Set fixed axis limits using pre-calculated global bounds
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_zlim([z_min, z_max])
        
        # Set labels and title with white color
        ax.set_xlabel('X (m)', color='white')
        ax.set_ylabel('Y (m)', color='white')
        ax.set_zlabel('Z (m)', color='white')
        ax.set_title(f'Body Frame - Session {session_swing}\nFrame {frame}/{len(markers.time)}', color='white')
        
        # Set tick colors to white
        ax.tick_params(colors='white')
        
        # Make grid and panes black with light grey gridlines
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('black')
        ax.yaxis.pane.set_edgecolor('black')
        ax.zaxis.pane.set_edgecolor('black')
        ax.grid(True, color='lightgrey', alpha=0.15)
        
        # Set viewing angle
        ax.view_init(elev=15, azim=35 if batter_hand == "L" else -35)
    
    # Create animation with every 10th frame to reduce file size
    frame_skip = 10
    frames = range(250, len(markers.time), frame_skip)
    
    print(f"Creating animation with {len(frames)} frames...")
    anim = animation.FuncAnimation(fig, animate, frames=frames, interval=100, repeat=True)
    
    # Apply tight layout to minimize padding
    plt.tight_layout(pad=0)
    
    # Save as GIF
    print(f"Saving GIF to {output_path}...")
    anim.save(output_path, writer='pillow', fps=10, dpi=100)
    print(f"GIF saved successfully!")
    
    plt.close()

if __name__ == "__main__":
    # Create the visualization
    visualize_body_frame(session_swing="447_5", output_path="./viz/body_frame.gif")