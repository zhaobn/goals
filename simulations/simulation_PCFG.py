from itertools import product
import pandas as pd
import numpy as np
from tqdm import tqdm
from rllib.shapeworld import State, Shape, ShapeWorld
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

productions = [
  ['S', ['and(S,S)', 'A']], # allow conjunctions
  ['A', ['one(B)', 'two(B)', 'three(B)', # location symmetry, fix one feature (location agnostic, compare feature)
         'two(E)', 'three(E)', # location symetry, feature symmetry (location agnostic, feature agnostic)
         'two(F,E)', 'three(F,E)', # location symmetry, multiple feature symetry (location agnostic, feature agnostic)
         'one(B,C)', 'two(B,D)', # feature in location, specify feature (location given, feature compare)
         'two(E,D)'# give location, feature symmetry (location given, feature agnostic)
         ]],
  ['B', ['square', 'circle', 'triangle', 'low', 'medium', 'high', 'striped', 'plain']], # features
  ['C', ['(0)', '(1)', '(2)']], # locations
  ['D', ['(0,1)', '(0,2)', '(1,2)']], # diad locations
  ['E', ['same','unique']], # relation,
  ['F', ['1','2','3']], # number of features to compare
]

def generate_all_programs(productions, start_symbol='S', max_depth=3):
    """
    Generate all possible programs from a PCFG up to a specified depth.
    
    Returns:
        dict: Mapping from programs to their log probabilities
    """
    def expand(symbol, depth):
        '''Take a symbol, and expand with log probabilities.'''
        if depth > max_depth:
            return {}
            
        # If symbol is terminal
        if not any(p[0] == symbol for p in productions):
            return {symbol: 0.0}  # log(1) = 0 for terminals
            
        # Find all productions for this symbol
        results = {}
        # Count number of productions for this symbol for probability calculation
        num_productions = sum(len(rhs_list) for lhs, rhs_list in productions if lhs == symbol)
        # Log probability of choosing each production
        log_prod_prob = -np.log(num_productions)
        
        for lhs, rhs_list in productions:
            if lhs == symbol:
                for rhs in rhs_list:
                    if '(' in rhs:
                        # Split into function and arguments
                        func, args = rhs.split('(')
                        args = args[:-1]  # remove closing parenthesis
                        arg_symbols = args.split(',')
                        
                        # Recursively expand each argument
                        arg_expansions = [expand(arg.strip(), depth + 1) for arg in arg_symbols]
                        
                        # Generate all combinations with their log probabilities
                        for arg_combo in product(*[exp.items() for exp in arg_expansions]):
                            args_strs, log_probs = zip(*arg_combo)
                            prog = f"{func}({','.join(args_strs)})"
                            # Sum log probabilities (equivalent to multiplying probabilities)
                            log_prob = log_prod_prob + sum(log_probs)
                            results[prog] = log_prob
                    else:
                        # Direct expansion
                        expansions = expand(rhs, depth + 1)
                        for exp, log_p in expansions.items():
                            results[exp] = log_prod_prob + log_p
                        
        return results

    # Generate programs with log probabilities
    program_log_probs = expand(start_symbol, 0)
    # Add quotes and return log probabilities
    return {f'"{prog}"': log_prob for prog, log_prob in program_log_probs.items()}

def generate_state_space():
    """Generate all possible states in ShapeWorld."""
    # Get all possible shapes
    shapes = [
        Shape(sides=sides, shade=shade, texture=texture)
        for sides in ['circle', 'square', 'triangle']
        for shade in ['low', 'medium', 'high']
        for texture in ['plain', 'striped', 'dots']
    ]
    
    # Generate all possible states (combinations of 3 shapes)
    return [State(shape1=s1, shape2=s2, shape3=s3) 
            for s1, s2, s3 in product(shapes, repeat=3)]

def program_applies(program: str, state: State) -> bool:
    """Check if a program applies to a state."""
    # Implementation from before - let me know if you need this implemented
    pass

def calculate_state_probabilities(programs_df: pd.DataFrame, states: list[State]) -> pd.DataFrame:
    """Calculate selection probability for each state.
    
    For each program:
    1. Find states it applies to
    2. Add log prob - log(num_matching_states) to those states' total log probs
    """
    logger.info("Calculating state probabilities...")
    
    # Initialize log probabilities for each state
    state_log_probs = {state: float('-inf') for state in states}  # log(0)
    
    # For each program
    for _, row in tqdm(programs_df.iterrows(), total=len(programs_df)):
        program = row['program']
        program_log_prob = row['log_probability']
        
        # Find matching states
        matching_states = [s for s in states if program_applies(program, s)]
        
        if matching_states:  # Only if program matches any states
            # Log probability of selecting any one matching state
            state_log_prob = program_log_prob - np.log(len(matching_states))
            
            # Add to each matching state's probability (log space addition)
            for state in matching_states:
                state_log_probs[state] = np.logaddexp(
                    state_log_probs[state],
                    state_log_prob
                )
    
    # Convert to DataFrame
    states_df = pd.DataFrame({
        'state': list(state_log_probs.keys()),
        'log_probability': list(state_log_probs.values())
    }).sort_values('log_probability', ascending=False)
    
    return states_df

def main():
    programs_file = Path('pcfg_programs.csv')
    states_file = Path('state_probabilities.csv')
    
    # Generate or load programs
    if not programs_file.exists():
        logger.info("Generating programs...")
        program_log_probs = generate_all_programs(productions, max_depth=3)
        
        programs_df = pd.DataFrame({
            'program': [prog.strip('"') for prog in program_log_probs.keys()],
            'log_probability': [program_log_probs[prog] for prog in program_log_probs.keys()]
        }).sort_values('log_probability', ascending=False)
        
        programs_df.to_csv(programs_file, index=False)
    else:
        logger.info("Loading existing programs...")
        programs_df = pd.read_csv(programs_file)
    
    # Generate state space
    logger.info("Generating state space...")
    states = generate_state_space()
    logger.info(f"Generated {len(states)} states")
    
    # Calculate state probabilities
    states_df = calculate_state_probabilities(programs_df, states)
    
    # Save results
    logger.info("Saving state probabilities...")
    states_df.to_csv(states_file, index=False)
    
    # Show sample of most likely states
    logger.info("\nMost likely states:")
    for _, row in states_df.head().iterrows():
        state = row['state']
        log_prob = row['log_probability']
        print(f"  {state} (log prob: {log_prob:.2f})")
    
    logger.info("Done!")

if __name__ == "__main__":
    main()

