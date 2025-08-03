import pandas as pd
import matplotlib.pyplot as plt

# Set font to Courier New
plt.rcParams['font.family'] = 'Courier New'

# Read and filter data
df = pd.read_csv('data/results_agged.csv')
filtered_df = df[
    (df['metric'] == 'median_absolute_error') & 
    (df['value'] > 0)
].copy()

# Round values and rename columns
filtered_df['median_absolute_error'] = filtered_df['value'].round(4)
filtered_df = filtered_df.rename(columns={'col': 'joint_angle'})

# Sort by median absolute error (descending for plotting - lowest at top)
filtered_df = filtered_df.sort_values('median_absolute_error', ascending=False)

# Create color mapping for joint types (pastel colors)
def get_joint_color(joint_name):
    joint_base = joint_name.rsplit('_', 1)[0] if '_' in joint_name else joint_name
    colors = {
        'hip': '#A8D8A8',      # Light Green
        'knee': '#87CEEB',     # Sky Blue  
        'ankle': '#DEB887',    # Burlywood
        'shoulder': '#FFB6C1', # Light Pink
        'elbow': '#DDA0DD',    # Plum
        'wrist': '#F0A0A0',    # Light Coral
    }
    
    for joint_type, color in colors.items():
        if joint_type in joint_base.lower():
            return color
    return '#C0C0C0'  # Light Gray for unknown joints

def get_joint_type(joint_name):
    joint_base = joint_name.rsplit('_', 1)[0] if '_' in joint_name else joint_name
    joint_types = ['hip', 'knee', 'ankle', 'shoulder', 'elbow', 'wrist']
    
    for joint_type in joint_types:
        if joint_type in joint_base.lower():
            return joint_type
    return 'other'

colors = [get_joint_color(joint) for joint in filtered_df['joint_angle']]
joint_types = [get_joint_type(joint) for joint in filtered_df['joint_angle']]

# Create horizontal bar chart
fig, ax = plt.subplots(figsize=(12, 8))
bars = ax.barh(filtered_df['joint_angle'], filtered_df['median_absolute_error'], 
               color=colors, edgecolor='white', linewidth=0.5)

ax.set_xlabel('Median Absolute Error', fontsize=16, fontweight='bold')
ax.set_ylabel('Joint Angle', fontsize=16, fontweight='bold')
ax.set_title('Median Absolute Error by Joint Angle', fontsize=14, fontweight='bold', pad=20)

# Style the plot
ax.grid(axis='x', alpha=0.7, linestyle='--', color='gray')
ax.set_axisbelow(False)
ax.tick_params(axis='both', which='major', labelsize=14)

# Create legend
unique_joints = list(set(joint_types))
legend_colors = {joint: get_joint_color(f'{joint}_x') for joint in unique_joints}
legend_patches = [plt.Rectangle((0,0),1,1, facecolor=color, label=joint.title()) 
                 for joint, color in legend_colors.items()]
ax.legend(handles=legend_patches, loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=16)

plt.tight_layout()


plt.savefig('viz/median_absolute_error.png', dpi=300, bbox_inches='tight')