import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import subprocess
from typing import List, Dict, Tuple
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def analyze_goal_state(state_str: str, programs_file: str) -> Dict:
    """Analyze a single goal state using state_coverage.py
    
    Args:
        state_str: State string in format: shape1_sides,shape1_shade,shape1_texture,...
        programs_file: Path to programs CSV file
        
    Returns:
        Dictionary with analysis results
    """
    try:
        # Get the absolute path to state_coverage.py
        state_coverage_path = os.path.abspath('state_coverage.py')
        
        result = subprocess.run(
            ['python', state_coverage_path, 
             '--programs', programs_file,
             '--state', state_str],
            capture_output=True,
            text=True
        )
        
        # Parse output to get minimum length
        output = result.stdout
        min_length = None
        for line in output.split('\n'):
            if 'Shortest Program:' in line:
                # Get the length from the next line
                length_line = output.split('\n')[output.split('\n').index(line) + 2]
                min_length = int(length_line.split(': ')[1])
                break
        
        return {
            'min_program_length': min_length,
            'state': state_str
        }
    except Exception as e:
        logger.error(f"Error analyzing state {state_str}: {str(e)}")
        return {
            'min_program_length': None,
            'state': state_str
        }

def analyze_all_goals(goals_file: str, programs_file: str, output_dir: str) -> pd.DataFrame:
    """Analyze minimum program lengths for all goals.
    
    Args:
        goals_file: Path to CSV file containing selected goals
        programs_file: Path to CSV file containing PCFG programs
        output_dir: Directory to save results
        
    Returns:
        DataFrame with analysis results
    """
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load selected goals
    logger.info(f"Loading goals from {goals_file}")
    selected_goals = pd.read_csv(goals_file)
    
    # Analyze each goal
    min_lengths = []
    for idx, row in selected_goals.iterrows():
        logger.info(f"Analyzing goal {idx + 1}/{len(selected_goals)}")
        
        # Convert goal to state string format
        state_str = f"{row['shape_0_type']},{row['shape_0_shade']},{row['shape_0_texture']}," \
                    f"{row['shape_1_type']},{row['shape_1_shade']},{row['shape_1_texture']}," \
                    f"{row['shape_2_type']},{row['shape_2_shade']},{row['shape_2_texture']}"
        
        # Analyze state
        result = analyze_goal_state(state_str, programs_file)
        result.update({
            'participant_id': row['participant_id'],
            'trial': row['trial_number']
        })
        min_lengths.append(result)
    
    # Convert to DataFrame
    min_lengths_df = pd.DataFrame(min_lengths)
    
    # Save results
    output_file = Path(output_dir) / 'min_program_lengths.csv'
    logger.info(f"Saving results to {output_file}")
    min_lengths_df.to_csv(output_file, index=False)
    
    # Create and save plots
    create_plots(min_lengths_df, output_dir)
    
    return min_lengths_df

def create_plots(min_lengths_df: pd.DataFrame, output_dir: str):
    """Create and save plots of program length analysis.
    
    Args:
        min_lengths_df: DataFrame with program length analysis
        output_dir: Directory to save plots
    """
    # Distribution of minimum program lengths
    plt.figure(figsize=(10, 6))
    sns.histplot(data=min_lengths_df, x='min_program_length', bins=20)
    plt.title('Distribution of Minimum Program Lengths')
    plt.xlabel('Minimum Program Length')
    plt.ylabel('Count')
    plt.savefig(Path(output_dir) / 'min_program_lengths_distribution.png')
    plt.close()
    
    # Average minimum length by participant
    participant_avg_lengths = min_lengths_df.groupby('participant_id')['min_program_length'].mean()
    plt.figure(figsize=(12, 6))
    participant_avg_lengths.plot(kind='bar')
    plt.title('Average Minimum Program Length by Participant')
    plt.xlabel('Participant ID')
    plt.ylabel('Average Minimum Program Length')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'avg_min_program_length_by_participant.png')
    plt.close()
    
    # Print summary statistics
    print("Summary of Minimum Program Lengths:")
    print(f"Mean: {min_lengths_df['min_program_length'].mean():.2f}")
    print(f"Median: {min_lengths_df['min_program_length'].median():.2f}")
    print(f"Min: {min_lengths_df['min_program_length'].min()}")
    print(f"Max: {min_lengths_df['min_program_length'].max()}")
    print(f"Std: {min_lengths_df['min_program_length'].std():.2f}")
    
    print("\nAverage Minimum Program Length by Participant:")
    print(participant_avg_lengths)

def main():
    """Main function to run the analysis."""
    goals_file = 'data-processed/selected_goals_with_nll.csv'
    programs_file = '../simulations/pcfg_programs.csv'
    output_dir = 'analysis/program_lengths'
    
    analyze_all_goals(goals_file, programs_file, output_dir)

if __name__ == '__main__':
    main() 