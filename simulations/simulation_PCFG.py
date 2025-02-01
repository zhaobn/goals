from itertools import product
import pandas as pd
import numpy as np
from tqdm import tqdm
from rllib.shapeworld import State, Shape
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

def main():
    output_file = Path('pcfg_programs.csv')
    
    # Check if programs file already exists
    if output_file.exists():
        logger.info(f"Programs file already exists at {output_file}")
        # Load a few programs to display
        df = pd.read_csv(output_file, nrows=5)
        logger.info("Sample of existing programs:")
        for _, row in df.iterrows():
            print(f"  {row['program']} (log prob: {row['log_probability']:.2f})")
        return
    
    # Generate programs if file doesn't exist
    logger.info("Generating programs...")
    program_log_probs = generate_all_programs(productions, max_depth=3)
    
    # Convert to DataFrame and sort by log probability (descending)
    logger.info(f"Saving {len(program_log_probs)} programs...")
    df = pd.DataFrame({
        'program': [prog.strip('"') for prog in program_log_probs.keys()],
        'log_probability': [program_log_probs[prog] for prog in program_log_probs.keys()]
    }).sort_values('log_probability', ascending=False)
    
    # Save with efficient compression and chunking
    df.to_csv(output_file, 
              index=False,
              compression='gzip' if len(program_log_probs) > 1_000_000 else None,
              chunksize=100_000,
              quoting=1)
    
    logger.info("Sample of most likely programs:")
    for _, row in df.head().iterrows():
        print(f"  {row['program']} (log prob: {row['log_probability']:.2f})")
    
    logger.info("Done!")

if __name__ == "__main__":
    main()

