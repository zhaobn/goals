import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

def load_data():
    """Load the selected goals data."""
    return pd.read_csv('./data-processed/selected_goals.csv')

def create_shape(ax, shape_type, shade, texture, position):
    """Create a shape with the specified properties."""
    x, y = position
    size = 0.8
    
    # Map shade to color intensity
    if shade == 'low':
        color_intensity = 0.1
    elif shade == 'medium':
        color_intensity = 0.6
    else:  # high
        color_intensity = 0.9
    
    color = (0, 0, color_intensity)  # Blue with varying intensity
    
    # Create the shape
    if shape_type == 'circle':
        shape = patches.Circle((x, y), size/2, fill=True, color=color, alpha=0.8)
    elif shape_type == 'triangle':
        shape = patches.RegularPolygon((x, y), numVertices=3, radius=size/2, 
                                      color=color, alpha=0.8)
    else:  # square
        shape = patches.Rectangle((x - size/2, y - size/2), size, size, fill=True, color=color, alpha=0.8)
    
    ax.add_patch(shape)
    
    # Add texture
    if texture == 'dots':
        # Add dots
        for _ in range(5):
            dot_x = x + np.random.uniform(-size/3, size/3)
            dot_y = y + np.random.uniform(-size/3, size/3)
            dot = patches.Circle((dot_x, dot_y), size/10, fill=True, color='white', alpha=0.7)
            ax.add_patch(dot)
    elif texture == 'stripes':
        # Add stripes
        for i in range(-2, 3):
            line = plt.Line2D([x - size/2, x + size/2], 
                              [y + i*size/5, y + i*size/5], 
                              lw=2, color='white', alpha=0.7)
            ax.add_line(line)

def visualize_goal(goal_row, ax=None, title=None):
    """Visualize a single goal configuration."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 2))
    
    # Set up the axes
    ax.set_xlim(-3, 3)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Add title if provided
    if title:
        ax.set_title(title)
    
    # Create the three shapes
    positions = [(-2, 0), (0, 0), (2, 0)]
    
    for i in range(3):
        shape_type = goal_row[f'shape_{i}_type']
        shade = goal_row[f'shape_{i}_shade']
        texture = goal_row[f'shape_{i}_texture']
        
        create_shape(ax, shape_type, shade, texture, positions[i])
    
    return ax

def visualize_participant_goals(data, participant_id, max_goals=20, save_path=None):
    """Visualize all goals selected by a participant in a single image."""
    participant_data = data[data['participant_id'] == participant_id]
    
    # Sort by trial number
    participant_data = participant_data.sort_values('trial_number')
    
    # Limit the number of goals to display
    if len(participant_data) > max_goals:
        participant_data = participant_data.iloc[:max_goals]
    
    n_goals = len(participant_data)
    
    # Calculate grid dimensions
    n_cols = min(5, n_goals)  # Maximum 5 columns
    n_rows = (n_goals + n_cols - 1) // n_cols  # Ceiling division
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 2*n_rows))
    
    # Flatten axes array for easy indexing
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()
    
    # Create each goal visualization
    for i, (_, goal) in enumerate(participant_data.iterrows()):
        title = f"Trial {goal['trial_number']}"
        if goal['abandoned']:
            title += " (Abandoned)"
        elif goal['goal_achieved']:
            title += " (Achieved)"
        else:
            title += " (Not Achieved)"
            
        visualize_goal(goal, axes[i], title)
    
    # Hide unused subplots
    for i in range(n_goals, len(axes)):
        axes[i].axis('off')
    
    # Add overall title
    plt.suptitle(f"Goals Selected by Participant: {participant_id}", fontsize=16, y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for suptitle
    
    # Create directory if it doesn't exist
    os.makedirs('./figures/goals', exist_ok=True)
    
    # Save figure
    if save_path is None:
        save_path = f'./figures/goals/participant_{participant_id}.png'
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def visualize_all_participants(data, max_participants=None, max_goals_per_participant=20):
    """Visualize goals for all participants."""
    participant_ids = data['participant_id'].unique()
    
    if max_participants and len(participant_ids) > max_participants:
        participant_ids = participant_ids[:max_participants]
    
    for participant_id in participant_ids:
        print(f"Visualizing goals for participant: {participant_id}")
        visualize_participant_goals(data, participant_id, max_goals_per_participant)

def main():
    # Load data
    data = load_data()
    
    # Create output directories
    os.makedirs('./figures', exist_ok=True)
    os.makedirs('./figures/goals', exist_ok=True)
    
    # Visualize goals for all participants
    print("Generating visualizations for all participants...")
    visualize_all_participants(data)
    
    # Also create a sample visualization of a single goal
    sample_goal = data.iloc[0]
    fig, ax = plt.subplots(figsize=(6, 2))
    visualize_goal(sample_goal, ax, f"Sample Goal (Participant: {sample_goal['participant_id']}, Trial: {sample_goal['trial_number']})")
    plt.tight_layout()
    plt.savefig('./figures/sample_goal.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("All visualizations have been saved to the './figures/goals/' directory.")

if __name__ == "__main__":
    main() 