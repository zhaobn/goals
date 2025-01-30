import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from rllib.shapeworld import State, Shape
from pathlib import Path
import logging
from typing import Dict, Tuple
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Assuming the same State namedtuple definition is available
# from collections import namedtuple
# Shape = namedtuple('Shape',['sides', 'shade', 'texture'])
# State = namedtuple('State',['shape1', 'shape2', 'shape3'])

def state_to_dict(state: State) -> dict:
    """Convert a State object to a dictionary for CSV output.
    
    Args:
        state: State object to convert
        
    Returns:
        Dictionary with flattened state attributes
    """
    return {
        'shape1_sides': state.shape1.sides,
        'shape1_shade': state.shape1.shade,
        'shape1_texture': state.shape1.texture,
        'shape2_sides': state.shape2.sides,
        'shape2_shade': state.shape2.shade,
        'shape2_texture': state.shape2.texture,
        'shape3_sides': state.shape3.sides,
        'shape3_shade': state.shape3.shade,
        'shape3_texture': state.shape3.texture
    }

def load_summary_statistics(directory: str) -> Tuple[Dict[State, float], Dict[State, float]]:
    """Load mean and median values from CSV files in the directory.
    
    Args:
        directory: Path to directory containing value function CSV files
        
    Returns:
        Tuple of two dictionaries: (mean_values, median_values) mapping goal states to their values
        
    Raises:
        ValueError: If any files failed to load
    """
    directory_path = Path(directory)
    if not directory_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    logger.info(f'Loading statistics from directory: {directory}')
    csv_files = list(directory_path.glob('value_function_goal_*.csv'))
    
    if not csv_files:
        raise ValueError(f"No value function CSV files found in {directory}")
    
    # Pre-allocate dictionaries with expected size
    n_files = len(csv_files)
    mean_values = dict.fromkeys(range(n_files))
    median_values = dict.fromkeys(range(n_files))
    
    logger.info(f"Processing {n_files} files...")
    
    failed_files = []
    for i, filepath in enumerate(tqdm(csv_files, desc='Loading files')):
        try:
            # Read only the necessary columns for goal state identification
            goal_cols = ['is_goal', 'shape1_sides', 'shape1_shade', 'shape1_texture',
                        'shape2_sides', 'shape2_shade', 'shape2_texture',
                        'shape3_sides', 'shape3_shade', 'shape3_texture']
            
            # First read to find goal state
            goal_df = pd.read_csv(filepath, usecols=goal_cols)
            goal_row = goal_df[goal_df['is_goal']].iloc[0]
            
            goal_state = State(
                shape1=Shape(sides=goal_row['shape1_sides'], 
                           shade=goal_row['shape1_shade'], 
                           texture=goal_row['shape1_texture']),
                shape2=Shape(sides=goal_row['shape2_sides'], 
                           shade=goal_row['shape2_shade'], 
                           texture=goal_row['shape2_texture']),
                shape3=Shape(sides=goal_row['shape3_sides'], 
                           shade=goal_row['shape3_shade'], 
                           texture=goal_row['shape3_texture'])
            )
            
            # Read only summary rows for statistics
            summary_df = pd.read_csv(filepath, 
                                   usecols=['shape1_sides', 'metric', 'value'])
            summary_rows = summary_df[summary_df['shape1_sides'] == 'SUMMARY']
            
            mean_value = float(summary_rows[summary_rows['metric'] == 'mean_value']['value'].iloc[0])
            median_value = float(summary_rows[summary_rows['metric'] == 'median_value']['value'].iloc[0])
            
            mean_values[goal_state] = mean_value
            median_values[goal_state] = median_value
            
        except Exception as e:
            logger.error(f"Error loading {filepath}: {str(e)}")
            failed_files.append(filepath)
            continue
    
    # Remove any None values that weren't filled due to errors
    mean_values = {k: v for k, v in mean_values.items() if v is not None}
    median_values = {k: v for k, v in median_values.items() if v is not None}
    
    # Check if any files failed to load
    if failed_files:
        error_msg = f"Failed to load {len(failed_files)} files:\n" + "\n".join(str(f) for f in failed_files)
        raise ValueError(error_msg)
            
    logger.info(f"Loaded statistics for {len(mean_values)} goals")
    return mean_values, median_values

def save_value_function(value_dict: Dict[State, float], output_file: str, metric_name: str) -> None:
    """Save the value function as a CSV file.
    
    Args:
        value_dict: Dictionary mapping goal states to their values
        output_file: Path to save CSV file
        metric_name: Name of the metric (mean or median)
    """
    logger.info(f"Saving {metric_name} values to {output_file}")
    try:
        # Create DataFrame with pre-allocated size
        n_states = len(value_dict)
        
        # Sort states by value before creating DataFrame
        sorted_items = sorted(value_dict.items(), key=lambda x: x[1], reverse=True)
        
        df = pd.DataFrame({
            'shape1_sides': [''] * n_states,
            'shape1_shade': [''] * n_states,
            'shape1_texture': [''] * n_states,
            'shape2_sides': [''] * n_states,
            'shape2_shade': [''] * n_states,
            'shape2_texture': [''] * n_states,
            'shape3_sides': [''] * n_states,
            'shape3_shade': [''] * n_states,
            'shape3_texture': [''] * n_states,
            'value': np.zeros(n_states),
            'metric': np.full(n_states, np.nan),
            'metric_value': np.full(n_states, np.nan)
        })
        
        # Fill the DataFrame with sorted values
        for i, (state, value) in enumerate(sorted_items):
            state_dict = state_to_dict(state)
            for key, val in state_dict.items():
                df.loc[i, key] = val
            df.loc[i, 'value'] = value
        
        # Add summary statistics
        summary_stats = {
            'total_states': len(df),
            f'{metric_name}_value': df['value'].mean(),
            'max_value': df['value'].max(),
            'min_value': df['value'].min(),
            'std_value': df['value'].std()
        }
        
        # Create summary rows DataFrame
        summary_df = pd.DataFrame([{
            'shape1_sides': 'SUMMARY',
            'shape1_shade': 'SUMMARY_ROW',
            'shape1_texture': 'SUMMARY_ROW',
            'shape2_sides': 'SUMMARY_ROW',
            'shape2_shade': 'SUMMARY_ROW',
            'shape2_texture': 'SUMMARY_ROW',
            'shape3_sides': 'SUMMARY_ROW',
            'shape3_shade': 'SUMMARY_ROW',
            'shape3_texture': 'SUMMARY_ROW',
            'value': value,
            'metric': metric,
            'metric_value': str(value)
        } for metric, value in summary_stats.items()])
        
        # Put summary at top, followed by sorted states
        df = pd.concat([summary_df, df], ignore_index=True)
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        logger.info(f"Successfully saved {metric_name} values with {len(df)} rows")
        
    except Exception as e:
        raise IOError(f"Failed to save CSV: {str(e)}")

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Aggregate value functions from individual goal files.')
    parser.add_argument('--input-dir', type=str, default='./value-iteration-results',
                       help='Directory containing value function CSV files (default: ./value-iteration-results)')
    parser.add_argument('--output-dir', type=str, default='./value-functions',
                       help='Directory to save aggregated results (default: ./value-functions)')
    
    args = parser.parse_args()
    
    try:
        # Create output directory if it doesn't exist
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Load mean and median values from CSV files
        mean_values, median_values = load_summary_statistics(args.input_dir)

        # Save mean values
        mean_output = output_dir / 'goal_value_function_mean.csv'
        save_value_function(mean_values, mean_output, 'mean')

        # Save median values
        median_output = output_dir / 'goal_value_function_median.csv'
        save_value_function(median_values, median_output, 'median')
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()


