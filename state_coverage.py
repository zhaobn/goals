import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import argparse
import logging
from pathlib import Path
from rllib.shapeworld import State, Shape
from simulation_PCFG import program_applies, productions
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def count_program_primitives(program: str) -> Dict[str, int]:
    """Count the number of primitive features in a program.
    
    Args:
        program: The program string to analyze
        
    Returns:
        Dictionary with counts for each primitive feature
    """
    # Initialize counters
    counts = {
        'sides': {'circle': 0, 'square': 0, 'triangle': 0},
        'shade': {'low': 0, 'medium': 0, 'high': 0},
        'texture': {'plain': 0, 'stripes': 0, 'dots': 0},
        'locations': 0,
        'relations': 0,
        'features': 0
    }
    
    # Count primitive features
    for feature in ['circle', 'square', 'triangle']:
        counts['sides'][feature] = program.count(feature)
    for shade in ['low', 'medium', 'high']:
        counts['shade'][shade] = program.count(shade)
    for texture in ['plain', 'stripes', 'dots']:
        counts['texture'][texture] = program.count(texture)
    
    # Count locations (numbers in parentheses)
    counts['locations'] = len(re.findall(r'\(\d+\)', program))
    
    # Count relations
    counts['relations'] = program.count('same') + program.count('unique')
    
    # Count total features
    counts['features'] = sum(sum(counts['sides'].values()) + 
                           sum(counts['shade'].values()) + 
                           sum(counts['texture'].values()))
    
    return counts

def get_applicable_programs(state: State, programs_df: pd.DataFrame) -> pd.DataFrame:
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
            # Count primitives in the program
            primitive_counts = count_program_primitives(program)
            # Calculate program length (number of tokens)
            program_length = len(program.split())
            
            # Add counts to the row
            row_dict = row.to_dict()
            row_dict.update(primitive_counts)
            row_dict['program_length'] = program_length
            applicable_programs.append(row_dict)
    
    return pd.DataFrame(applicable_programs)

def analyze_state(state: State, programs_df: pd.DataFrame) -> Dict:
    """Analyze a state's coverage and program primitives.
    
    Args:
        state: The state to analyze
        programs_df: DataFrame containing programs and their probabilities
        
    Returns:
        Dictionary containing analysis results
    """
    # Get applicable programs
    applicable_programs = get_applicable_programs(state, programs_df)
    
    # Calculate coverage statistics
    total_programs = len(programs_df)
    covered_programs = len(applicable_programs)
    coverage_percentage = (covered_programs / total_programs) * 100
    
    # Find shortest program
    if not applicable_programs.empty:
        shortest_program = applicable_programs.loc[applicable_programs['program_length'].idxmin()]
    else:
        shortest_program = None
    
    # Calculate average primitive counts
    avg_primitives = {
        'sides': {k: applicable_programs[k].mean() for k in ['circle', 'square', 'triangle']},
        'shade': {k: applicable_programs[k].mean() for k in ['low', 'medium', 'high']},
        'texture': {k: applicable_programs[k].mean() for k in ['plain', 'stripes', 'dots']},
        'locations': applicable_programs['locations'].mean(),
        'relations': applicable_programs['relations'].mean(),
        'features': applicable_programs['features'].mean()
    }
    
    return {
        'total_programs': total_programs,
        'covered_programs': covered_programs,
        'coverage_percentage': coverage_percentage,
        'shortest_program': shortest_program,
        'avg_primitives': avg_primitives,
        'applicable_programs': applicable_programs
    }

def main():
    parser = argparse.ArgumentParser(description='Analyze state coverage and program primitives')
    parser.add_argument('--programs', type=str, required=True, help='Path to programs CSV file')
    parser.add_argument('--state', type=str, required=True, 
                       help='State to analyze in format: shape1_sides,shape1_shade,shape1_texture,shape2_sides,shape2_shade,shape2_texture,shape3_sides,shape3_shade,shape3_texture')
    
    args = parser.parse_args()
    
    # Load programs
    logger.info(f"Loading programs from {args.programs}")
    programs_df = pd.read_csv(args.programs)
    
    # Parse state
    state_parts = args.state.split(',')
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
    
    # Analyze state
    logger.info("Analyzing state...")
    results = analyze_state(state, programs_df)
    
    # Print results
    print("\nProgram Coverage:")
    print(f"  Total programs: {results['total_programs']}")
    print(f"  Covered programs: {results['covered_programs']}")
    print(f"  Coverage percentage: {results['coverage_percentage']:.2f}%")
    
    if results['shortest_program'] is not None:
        print("\nShortest Program:")
        print(f"  Program: {results['shortest_program']['program']}")
        print(f"  Length: {results['shortest_program']['program_length']}")
        print(f"  Log probability: {results['shortest_program']['log_probability']:.2f}")
        print("\n  Primitive counts:")
        print(f"    Features: {results['shortest_program']['features']}")
        print(f"    Locations: {results['shortest_program']['locations']}")
        print(f"    Relations: {results['shortest_program']['relations']}")
    
    print("\nAverage Primitive Counts Across All Applicable Programs:")
    for category, counts in results['avg_primitives'].items():
        if isinstance(counts, dict):
            print(f"  {category}:")
            for value, count in counts.items():
                print(f"    {value}: {count:.2f}")
        else:
            print(f"  {category}: {counts:.2f}")
    
    print("\nTop 5 applicable programs (by length):")
    for _, row in results['applicable_programs'].sort_values('program_length').head().iterrows():
        print(f"  {row['program']} (length: {row['program_length']}, log prob: {row['log_probability']:.2f})")

if __name__ == '__main__':
    main() 