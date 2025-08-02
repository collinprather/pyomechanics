import kineticstoolkit.lab as ktk
import os
from dotenv import load_dotenv
from pathlib import Path
from pyomechanics.utils import generate_marker_graph, add_custom_markers, parse_c3d_file_path
from pyomechanics.body import scapula_right, upper_arm_right, parts
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def visualize_rear_shoulder(session_swing="447_5", output_path="rear_shoulder_visualization.gif"):
    """
    Visualize 3D body with focus on rear shoulder joint and orientation vectors
    
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
    
    # Create axis frames for scapula and upper arm
    for part in parts:
        axis_frames = part.create_axis_frames(markers)
        markers.data[part.axis_frames_name] = axis_frames
    
    print(f"Loaded {len(markers.data)} markers with {len(markers.time)} time points")
    
    # Define body frame markers for the plot (full body for context)
    body_frame_markers = ['RSHO', 'LSHO', 'elbow_r', 'elbow_l', 'wrist_r', 'wrist_l', 
                          'hip_r', 'hip_l', 'knee_r', 'knee_l', 'ankle_r', 'ankle_l',
                          'heel_r', 'heel_l', 'LTOE', 'RTOE', 'torso_m', 'RFHD', 'LFHD', 'RBHD', 'LBHD', 'scapula_r']
    
    # Filter to only markers that exist in the data
    available_body_markers = [m for m in body_frame_markers if m in markers.data]
    
    # Use original markers for axis bounds calculation (exclude axis frames we added)
    original_markers = [name for name in markers.data.keys() if not name.endswith('_frames')]
    
    # Calculate global bounds across all markers and all time points
    all_positions = []
    for marker_name in original_markers:
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
    
    # Define body frame connections (full body for context)
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
        ('RFHD', 'torso_m'), ('LFHD', 'torso_m'), ('RBHD', 'torso_m'), ('LBHD', 'torso_m'),
        # Scapula to shoulder
        ('scapula_r', 'RSHO')
    ]
    
    # Set up the figure with black background
    fig = plt.figure(figsize=(12, 9), facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')
    
    # Minimize margins around the plot
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    def draw_orientation_vectors(ax, frame_data, origin, scale=0.1, colors=['red', 'green', 'blue']):
        """Draw X, Y, Z orientation vectors from a given origin"""
        if np.isnan(frame_data).any() or np.isnan(origin).any():
            return
            
        # Extract the rotation matrix (first 3x3 of the 4x4 transformation matrix)
        rotation_matrix = frame_data[:3, :3]
        
        # Ensure origin is only 3D coordinates (x, y, z)
        origin_3d = origin[:3] if len(origin) > 3 else origin
        
        # Draw X, Y, Z axes
        for i, color in enumerate(colors):
            # Get the direction vector for this axis
            direction = rotation_matrix[:, i]
            end_point = origin_3d + direction * scale
            
            ax.plot([origin_3d[0], end_point[0]], 
                   [origin_3d[1], end_point[1]], 
                   [origin_3d[2], end_point[2]], 
                   color=color, linewidth=3, alpha=0.95)
    
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
                    # Highlight right shoulder connections
                    # if marker1 in ['RSHO', 'scapula_r'] or marker2 in ['RSHO', 'scapula_r']:
                    if (marker1 == 'hip_r' and marker2 == 'RSHO') or (marker1 == 'RSHO' and marker2 == 'elbow_r'):
                        ax.plot([data1[frame, 0], data2[frame, 0]], 
                               [data1[frame, 1], data2[frame, 1]], 
                               [data1[frame, 2], data2[frame, 2]], 
                               color='orange', alpha=0.7, linewidth=3)
                    else:
                        ax.plot([data1[frame, 0], data2[frame, 0]], 
                               [data1[frame, 1], data2[frame, 1]], 
                               [data1[frame, 2], data2[frame, 2]], 
                               color='tab:blue', alpha=0.7, linewidth=2)
        
        # Draw orientation vectors for right scapula
        if 'scapula_r_frames' in markers.data and 'scapula_r' in markers.data:
            scapula_frames = markers.data['scapula_r_frames'][frame]
            scapula_origin = markers.data['scapula_r'][frame]
            if not np.isnan(scapula_frames).any() and not np.isnan(scapula_origin).any():
                draw_orientation_vectors(ax, scapula_frames, scapula_origin, scale=0.15, 
                                       colors=['red', 'green', 'blue'])
        
        # Draw orientation vectors for right upper arm
        if 'upper_arm_r_frames' in markers.data and 'RSHO' in markers.data:
            upper_arm_frames = markers.data['upper_arm_r_frames'][frame]
            upper_arm_origin = markers.data['RSHO'][frame]
            if not np.isnan(upper_arm_frames).any() and not np.isnan(upper_arm_origin).any():
                draw_orientation_vectors(ax, upper_arm_frames, upper_arm_origin, scale=0.15, 
                                       colors=['red', 'green', 'blue'])
        
        # Set fixed axis limits using pre-calculated global bounds
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_zlim([z_min, z_max])
        
        # Set labels and title with white color
        ax.set_xlabel('X (m)', color='white')
        ax.set_ylabel('Y (m)', color='white')
        ax.set_zlabel('Z (m)', color='white')
        ax.set_title(f'Right Shoulder Focus - Session {session_swing}\nFrame {frame}/{len(markers.time)}', color='white')
        
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
        
        # Set viewing angle to match original body frame script
        ax.view_init(elev=15, azim=90+55 if batter_hand == "L" else -(90+55))
    
    # Create animation with every 10th frame to reduce file size
    frame_skip = 10
    frames = range(250, len(markers.time), frame_skip)
    
    print(f"Creating animation with {len(frames)} frames...")
    anim = animation.FuncAnimation(fig, animate, frames=frames, interval=100, repeat=True)
    
    # Apply tight layout to minimize padding
    plt.tight_layout(pad=-20.0)
    
    # Save as GIF
    print(f"Saving GIF to {output_path}...")
    anim.save(output_path, writer='pillow', fps=3, dpi=100)
    print(f"GIF saved successfully!")
    
    plt.close()

if __name__ == "__main__":
    # Create the visualization
    visualize_rear_shoulder(session_swing="447_5", output_path="./viz/rear_shoulder.gif")