import kineticstoolkit.lab as ktk
import os
from dotenv import load_dotenv
from pathlib import Path
from pyomechanics.utils import generate_marker_graph, add_custom_markers, parse_c3d_file_path
from pyomechanics.body import scapula_right, upper_arm_right, shoulder_joint_right, parts
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def visualize_rear_shoulder_joint_angles(session_swing="447_5", output_path="rear_shoulder_angles.gif"):
    """
    Visualize right shoulder joint angles over time as animated line plot
    
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
    
    # Create axis frames for all body parts
    for part in parts:
        axis_frames = part.create_axis_frames(markers)
        markers.data[part.axis_frames_name] = axis_frames
    
    print(f"Loaded {len(markers.data)} markers with {len(markers.time)} time points")
    
    # Calculate right shoulder joint angles
    shoulder_angles = shoulder_joint_right.angles(markers, batter_hand=batter_hand)
    
    print(f"Calculated shoulder joint angles with shape: {shoulder_angles.shape}")
    
    # Create time array (in seconds, assuming marker time is available)
    time_array = markers.time
    
    # Set up the figure with black background
    fig = plt.figure(figsize=(12, 9), facecolor='black')
    ax = fig.add_subplot(111, facecolor='black')
    
    # Minimize margins around the plot with more space for labels
    fig.subplots_adjust(left=0.2, right=0.95, top=0.8, bottom=0.25)
    
    # Calculate y-axis limits with some padding
    all_angles = shoulder_angles.flatten()
    valid_angles = all_angles[~np.isnan(all_angles)]
    if len(valid_angles) > 0:
        y_min = np.min(valid_angles) - 10
        y_max = np.max(valid_angles) + 10
    else:
        y_min, y_max = -180, 180
    
    def animate(frame_idx):
        ax.clear()
        ax.set_facecolor('black')
        
        # Current frame in the original time series
        current_frame = frame_idx  # frame_idx is now the actual frame number
        
        # Plot angle lines up to current frame
        end_idx = min(current_frame + 1, len(time_array))
        
        if end_idx > 0:
            # X angle - Horizontal Abduction (+) / Adduction (-)
            ax.plot(time_array[:end_idx], shoulder_angles[:end_idx, 0], 
                   color='red', linewidth=3, alpha=0.9, 
                   label='Horizontal Ab (+)/Add (-)')
            
            # Y angle - Abduction (+) / Adduction (-)
            ax.plot(time_array[:end_idx], shoulder_angles[:end_idx, 1], 
                   color='green', linewidth=3, alpha=0.9,
                   label='Ab (+)/Add (-)')
            
            # Z angle - External (+) / Internal (-) Rotation
            ax.plot(time_array[:end_idx], shoulder_angles[:end_idx, 2], 
                   color='blue', linewidth=3, alpha=0.9,
                   label='External (+)/Internal (-) Rotation')
            
            # Add current time marker
            if current_frame < len(time_array):
                current_time = time_array[current_frame]
                ax.axvline(x=current_time, color='white', linestyle='--', alpha=0.7, linewidth=2)
        
        # Set axis limits - x-axis starts from frame 250
        start_frame = 250
        ax.set_xlim([time_array[start_frame], time_array[-1]])
        ax.set_ylim([y_min, y_max])
        
        # Set labels and title with white color and larger fonts
        ax.set_xlabel('Time (s)', color='white', fontsize=20)
        ax.set_ylabel('Joint Angles (degrees)', color='white', fontsize=20)
        ax.set_title(f'Right Shoulder Joint Angles - Session {session_swing}\nTime: {time_array[min(current_frame, len(time_array)-1)]:.2f}s', 
                    color='white', fontsize=22)
        
        # Set tick colors to white with larger font
        ax.tick_params(colors='white', labelsize=16)
        
        # Add legend with white text and larger font in bottom left
        ax.legend(loc='lower left', frameon=True, facecolor='black', 
                 edgecolor='white', labelcolor='white', fontsize=24)
        
        # Set spines to white
        for spine in ax.spines.values():
            spine.set_color('white')
        
        # Add grid with light grey color
        ax.grid(True, color='lightgrey', alpha=0.3)
        
        # Add horizontal line at zero
        ax.axhline(y=0, color='white', linestyle='-', alpha=0.5, linewidth=1)
    
    # Create animation with every 10th frame to reduce file size, starting from frame 250
    frame_skip = 10
    frames = range(250, len(time_array), frame_skip)
    
    print(f"Creating animation with {len(frames)} frames...")
    anim = animation.FuncAnimation(fig, animate, frames=frames, interval=100, repeat=True)
    
    # Apply tight layout to minimize padding
    plt.tight_layout(pad=6.0)
    
    # Save as GIF
    print(f"Saving GIF to {output_path}...")
    anim.save(output_path, writer='pillow', fps=3, dpi=100)
    print(f"GIF saved successfully!")
    
    plt.close()

if __name__ == "__main__":
    # Create the visualization
    visualize_rear_shoulder_joint_angles(session_swing="447_5", output_path="./viz/rear_shoulder_joint_angles.gif")