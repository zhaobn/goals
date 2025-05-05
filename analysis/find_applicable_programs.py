import pandas as pd
import numpy as np
from pathlib import Path
import logging
from rllib.shapeworld import State, Shape
from simulation_PCFG import program_applies

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def find_applicable_programs(state: State, programs_df: pd.DataFrame) -> pd.DataFrame:
    """Find all programs that apply to a given state.
    
    Args:
        state: The state to check
        programs_df: DataFrame containing programs and their probabilities
        
    Returns:
        DataFrame with only the programs that apply to the state
    """
    applicable_programs = []
    
    for _, row in programs_df.iterrows():
        program = str(row['program'])
        if program_applies(program, state):
            # Add the program and its log probability
            applicable_programs.append({
                'program': program,
                'log_probability': row['log_probability']
            })
    
    return pd.DataFrame(applicable_programs)

def analyze_state(state_str: str, programs_file: str) -> pd.DataFrame:
    """Analyze a state and find all applicable programs.
    
    Args:
        state_str: State string in format: shape1_sides,shape1_shade,shape1_texture,...
        programs_file: Path to programs CSV file
        
    Returns:
        DataFrame with applicable programs
    """
    # Load programs
    logger.info(f"Loading programs from {programs_file}")
    programs_df = pd.read_csv(programs_file)
    
    # Parse state string into State object
    state_parts = state_str.split(',')
    if len(state_parts) != 9:
        raise ValueError("State must have 9 parts (3 shapes × 3 features)")
    
    state = State(
        shape1=Shape(
            sides=state_parts[0],
            shade=state_parts[1],
            texture=state_parts[2]
        ),
        shape2=Shape(
            sides=state_parts[3],
            shade=state_parts[4],
            texture=state_parts[5]
        ),
        shape3=Shape(
            sides=state_parts[6],
            shade=state_parts[7],
            texture=state_parts[8]
        )
    )
    
    # Find applicable programs
    logger.info("Finding applicable programs...")
    applicable_programs = find_applicable_programs(state, programs_df)
    
    # Sort by log probability (ascending = most likely first)
    applicable_programs = applicable_programs.sort_values('log_probability')
    
    # Print summary
    print(f"\nFound {len(applicable_programs)} applicable programs")
    print("\nTop 5 most likely programs:")
    for _, row in applicable_programs.head().iterrows():
        print(f"  {row['program']} (log prob: {row['log_probability']:.2f})")
    
    return applicable_programs

def main():
    """Example usage"""
    # Example state: circle,low,plain,square,medium,stripes,triangle,high,dots
    state_str = "circle,low,plain,square,medium,stripes,triangle,high,dots"
    programs_file = '../simulations/pcfg_programs.csv'
    
    applicable_programs = analyze_state(state_str, programs_file)
    
    # Save results
    output_file = 'analysis/program_lengths/applicable_programs.csv'
    Path('analysis/program_lengths').mkdir(parents=True, exist_ok=True)
    applicable_programs.to_csv(output_file, index=False)
    logger.info(f"Saved results to {output_file}")

if __name__ == '__main__':
    main() 