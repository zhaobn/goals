import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from rllib.shapeworld import State, Shape
from pathlib import Path
import logging
from typing import Dict, Tuple

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

def load_value_functions(directory: str) -> Dict[State, Dict[State, float]]:
    """Load all value function files from the specified directory.
    
    Args:
        directory: Path to directory containing value function files
        
    Returns:
        Dictionary mapping goal states to their value functions
        
    Raises:
        FileNotFoundError: If directory doesn't exist
        ValueError: If no .pkl files found in directory
    """
    directory_path = Path(directory)
    if not directory_path.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    logger.info(f'Loading value functions from directory: {directory}')
    value_functions = {}
    pkl_files = list(directory_path.glob('*.pkl'))
    
    if not pkl_files:
        raise ValueError(f"No .pkl files found in {directory}")
    
    for filepath in tqdm(pkl_files, desc='Loading files'):
        try:
            value_function, goal_state = pd.read_pickle(filepath)
            value_functions[goal_state] = value_function
        except Exception as e:
            logger.error(f"Error loading {filepath}: {str(e)}")
            continue
            
    logger.info(f"Loaded {len(value_functions)} value functions")
    return value_functions

def calculate_goal_value_functions(
    value_functions: Dict[State, Dict[State, float]]
) -> Dict[State, float]:
    """Calculate the average value for each goal state across all state values.
    
    Args:
        value_functions: Dictionary mapping goal states to their value functions
        
    Returns:
        Dictionary mapping goal states to their average values
    """
    logger.info("Calculating goal value functions...")
    goal_value_function = {}
    
    for goal_state, value_function in tqdm(value_functions.items(), desc='Calculating values'):
        # Convert values to numpy array for faster computation
        values = np.array(list(value_function.values()))
        average_value = np.mean(values)
        goal_value_function[goal_state] = average_value
        
    logger.info(f"Calculated values for {len(goal_value_function)} goals")
    return goal_value_function

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

def save_as_csv(goal_value_function: Dict[State, float], output_file: str) -> None:
    """Save the goal value function as a CSV file.
    
    Args:
        goal_value_function: Dictionary mapping states to values
        output_file: Path to save CSV file
        
    Raises:
        IOError: If unable to write to output file
    """
    logger.info(f"Saving CSV to {output_file}")
    try:
        # Convert to DataFrame in one go for better performance
        rows = [
            {**state_to_dict(state), 'value': value}
            for state, value in goal_value_function.items()
        ]
        df = pd.DataFrame(rows)
        
        # Add some useful statistics
        df['value_rank'] = df['value'].rank(method='dense')
        df['value_percentile'] = df['value'].rank(pct=True)
        
        # Save with compression for large files
        df.to_csv(output_file, index=False, compression='gzip')
        logger.info(f"Successfully saved CSV with {len(df)} rows")
        
        # Print some basic statistics
        logger.info("\nValue Statistics:")
        logger.info(f"Mean: {df['value'].mean():.4f}")
        logger.info(f"Std: {df['value'].std():.4f}")
        logger.info(f"Min: {df['value'].min():.4f}")
        logger.info(f"Max: {df['value'].max():.4f}")
        
    except Exception as e:
        raise IOError(f"Failed to save CSV: {str(e)}")

def main():
    """Main execution function."""
    try:
        # Create output directory if it doesn't exist
        output_dir = Path('./value-functions')
        output_dir.mkdir(exist_ok=True)
        
        # Load value functions
        directory = './value-iteration-results'
        value_functions = load_value_functions(directory)

        # Calculate the goal value function
        goal_value_function = calculate_goal_value_functions(value_functions)

        # Save as pickle file
        pkl_output = output_dir / 'goal_value_function.pkl'
        pd.to_pickle(goal_value_function, pkl_output)
        logger.info(f'Saved pickle to {pkl_output}')

        # Save as CSV file
        csv_output = output_dir / 'goal_value_function.csv'
        save_as_csv(goal_value_function, csv_output)
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()


