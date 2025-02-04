from itertools import product
import pandas as pd
import numpy as np
from tqdm import tqdm
from rllib.shapeworld import State, Shape, ShapeWorld
import logging
from pathlib import Path
import sys
from functools import reduce

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
  ['B', ['square', 'circle', 'triangle', 'low', 'medium', 'high', 'plain', 'stripes', 'dots']], # features
  ['C', ['(0)', '(1)', '(2)']], # locations
  ['D', ['(0,1)', '(0,2)', '(1,2)']], # diad locations
  ['E', ['same','unique']], # relation,
  ['F', ['1','2','3']], # number of features to compare
]

def generate_all_programs(productions, start_symbol='S', max_depth=4):
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
        for texture in ['plain', 'stripes', 'dots']
    ]
    
    # Generate all possible states (combinations of 3 shapes)
    return [State(shape1=s1, shape2=s2, shape3=s3) 
            for s1, s2, s3 in product(shapes, repeat=3)]

def program_applies(program: str, state: State) -> bool:
    """Check if a program applies to a state."""
    try:
        # Remove quotes if present
        program = program.strip('"')
        
        # Handle conjunctions
        if program.startswith('and('):
            # Validate and format
            if not program.endswith(')'):
                logger.error(f"Invalid conjunction format (missing closing parenthesis): {program}")
                return False
                
            # Split into subprograms more carefully
            inner = program[4:-1]  # Remove 'and(' and ')'
            paren_count = 0
            split_point = -1
            
            # Find the correct comma to split on
            for i, char in enumerate(inner):
                if char == '(':
                    paren_count += 1
                elif char == ')':
                    paren_count -= 1
                elif char == ',' and paren_count == 0:
                    split_point = i
                    break
            
            if split_point == -1:
                logger.error(f"Invalid conjunction format (no main comma found): {program}")
                return False
                
            subprograms = [inner[:split_point].strip(), inner[split_point+1:].strip()]
            return program_applies(subprograms[0], state) and program_applies(subprograms[1], state)
        
        # Validate basic program format
        if not ('(' in program and program.endswith(')')):
            logger.error(f"Invalid program format: {program}")
            return False
            
        # Parse function and arguments more carefully
        try:
            func, rest = program.split('(', 1)
            if not rest.endswith(')'):
                logger.error(f"Invalid program format (missing closing parenthesis): {program}")
                return False
                
            # Split arguments with proper parentheses handling
            args_str = rest[:-1]  # Remove closing parenthesis
            args = []
            current_arg = ''
            paren_count = 0
            
            for char in args_str:
                if char == '(' and paren_count == 0:
                    current_arg += char
                    paren_count += 1
                elif char == ')' and paren_count == 1:
                    current_arg += char
                    paren_count -= 1
                elif char == ',' and paren_count == 0:
                    args.append(current_arg.strip())
                    current_arg = ''
                else:
                    current_arg += char
                    if char == '(':
                        paren_count += 1
                    elif char == ')':
                        paren_count -= 1
            
            if current_arg:
                args.append(current_arg.strip())
                
            if not args:
                logger.error(f"No arguments found in program: {program}")
                return False
                
        except ValueError as e:
            logger.error(f"Failed to parse program {program}: {str(e)}")
            return False
        
        # Get shapes from state for easier access
        shapes = [state.shape1, state.shape2, state.shape3]
        
        # Helper function to count matching features
        def count_matching_feature(feature: str) -> int:
            if feature in ['square', 'circle', 'triangle']:
                return sum(1 for shape in shapes if shape.sides == feature)
            elif feature in ['low', 'medium', 'high']:
                return sum(1 for shape in shapes if shape.shade == feature)
            elif feature in ['plain', 'stripes', 'dots']:
                return sum(1 for shape in shapes if shape.texture == feature)
            return 0
        
        # Helper function to get shapes at specified locations
        def get_shapes_at_locations(loc_str: str) -> list:
            locations = [int(x) for x in loc_str.strip('()').split(',')]
            return [shapes[i] for i in locations]
        
        # Helper function to check if shapes have same feature
        def have_same_feature(shape_list: list) -> bool:
            return (len(set(s.sides for s in shape_list)) == 1 or
                    len(set(s.shade for s in shape_list)) == 1 or
                    len(set(s.texture for s in shape_list)) == 1)
        
        # Helper function to check if shapes have unique features
        def have_unique_features(shape_list: list) -> bool:
            return (len(set(s.sides for s in shape_list)) == len(shape_list) or
                    len(set(s.shade for s in shape_list)) == len(shape_list) or
                    len(set(s.texture for s in shape_list)) == len(shape_list))
        
        # Handle different program types
        if func == 'one':
            if len(args) == 1:  # one(feature)
                return count_matching_feature(args[0]) == 1
            else:  # one(feature,location)
                target_shapes = get_shapes_at_locations(args[1])
                return any(count_matching_feature(args[0]) == 1 for shape in target_shapes)
            
        elif func == 'two':
            if len(args) == 1:  # two(feature) or two(relation)
                if args[0] in ['same', 'unique']:
                    # Check pairs of shapes
                    pairs = [(shapes[i], shapes[j]) 
                            for i in range(3) for j in range(i+1, 3)]
                    if args[0] == 'same':
                        return any(have_same_feature([s1, s2]) for s1, s2 in pairs)
                    else:  # unique
                        return any(have_unique_features([s1, s2]) for s1, s2 in pairs)
                else:  # feature
                    return count_matching_feature(args[0]) == 2
                
            elif len(args) == 2:
                if args[0] in ['1', '2', '3']:  # two(num_features,relation)
                    num_features = int(args[0])
                    pairs = [(shapes[i], shapes[j]) 
                            for i in range(3) for j in range(i+1, 3)]
                    for s1, s2 in pairs:
                        matches = (
                            (s1.sides == s2.sides) +
                            (s1.shade == s2.shade) +
                            (s1.texture == s2.texture)
                        )
                        if matches == num_features:
                            return True
                    return False
                else:  # two(feature,location)
                    target_shapes = get_shapes_at_locations(args[1])
                    return all(count_matching_feature(args[0]) == 1 for shape in target_shapes)
        
        elif func == 'three':
            if len(args) == 1:
                if args[0] in ['same', 'unique']:
                    if args[0] == 'same':
                        return have_same_feature(shapes)
                    else:  # unique
                        return have_unique_features(shapes)
                else:  # feature
                    return count_matching_feature(args[0]) == 3
            elif len(args) == 2:  # three(num_features,relation)
                num_features = int(args[0])
                matches = (
                    (shapes[0].sides == shapes[1].sides == shapes[2].sides) +
                    (shapes[0].shade == shapes[1].shade == shapes[2].shade) +
                    (shapes[0].texture == shapes[1].texture == shapes[2].texture)
                )
                return matches == num_features
        
        return False

    except Exception as e:
        logger.error(f"Error processing program '{program}': {str(e)}")
        return False

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

def calculate_state_probabilities(programs_df: pd.DataFrame, states: list[State]) -> pd.DataFrame:
    """Calculate selection probability for each state."""
    logger.info("Calculating state probabilities...")
    
    # Initialize log probabilities for each state
    state_log_probs = {state: float('-inf') for state in states}  # log(0)
    
    # For each program
    problematic_programs = []
    for _, row in tqdm(programs_df.iterrows(), total=len(programs_df)):
        try:
            program = str(row['program'])  # Ensure string type
            program_log_prob = row['log_probability']
            
            # Find matching states
            matching_states = [s for s in states if program_applies(program, s)]
            
            if matching_states:  # Only if program matches any states
                # Log probability of selecting any one matching state (using log space division)
                state_log_prob = program_log_prob - np.log(len(matching_states))
                
                # Add to each matching state's probability (log space addition)
                for state in matching_states:
                    state_log_probs[state] = np.logaddexp(
                        state_log_probs[state],
                        state_log_prob
                    )
        except Exception as e:
            problematic_programs.append((program, str(e)))
            continue
    
    if problematic_programs:
        logger.warning(f"Found {len(problematic_programs)} problematic programs:")
        for prog, error in problematic_programs[:5]:  # Show first 5 problems
            logger.warning(f"  {prog}: {error}")
    
    # Normalize log probabilities
    log_probs = list(state_log_probs.values())
    # Compute log of sum of probabilities using log-sum-exp trick
    log_sum = reduce(np.logaddexp, log_probs)
    # Subtract log_sum from each log probability to normalize
    state_log_probs = {state: log_prob - log_sum 
                      for state, log_prob in state_log_probs.items()}
    
    # Convert to DataFrame with flattened state features
    results = []
    for state, log_prob in state_log_probs.items():
        state_dict = state_to_dict(state)
        state_dict['log_probability'] = log_prob
        results.append(state_dict)
    
    states_df = pd.DataFrame(results).sort_values('log_probability', ascending=False)
    
    # Verify normalization (optional)
    total_prob = np.sum(np.exp(states_df['log_probability']))
    logger.info(f"Sum of probabilities after normalization: {total_prob:.10f}")
    
    return states_df

def test_program_applies():
    """Test the program_applies function with various cases."""
    # Create some test states
    s1 = Shape(sides='square', shade='low', texture='plain')
    s2 = Shape(sides='circle', shade='low', texture='stripes')
    s3 = Shape(sides='triangle', shade='high', texture='dots')
    state1 = State(shape1=s1, shape2=s2, shape3=s3)
    
    # Test cases: (program, state, expected_result)
    test_cases = [
        # Basic feature counting
        ("one(square)", state1, True),
        ("two(low)", state1, True),
        ("three(plain)", state1, False),
        
        # Location-specific tests
        ('one(square,(0))', state1, True),
        ('one(circle,(1))', state1, True),
        ('one(square,(1))', state1, False),
        
        # Same/unique tests
        ('two(same)', state1, True),  # same shade (low)
        ('three(unique)', state1, True),  # all different sides
        
        # Multiple feature tests
        ('two(2,same)', state1, False),
        ('three(1,same)', state1, True),
        
        # Conjunction tests
        ('and(one(square),two(low))', state1, True),
        ('and(one(circle),one(triangle))', state1, True),
        ('and(two(plain),three(low))', state1, False),
    ]
    
    # Run tests
    failed_tests = []
    for program, state, expected in test_cases:
        result = program_applies(program, state)
        if result != expected:
            failed_tests.append(
                f"Failed: {program}\n"
                f"  State: {state}\n"
                f"  Expected: {expected}, Got: {result}"
            )
    
    # Report results
    if failed_tests:
        print("❌ Some tests failed:")
        for error in failed_tests:
            print(error)
            print()
    else:
        print("✅ All tests passed!")

def analyze_feature_distribution():
    """Analyze the distribution of features in generated programs."""
    programs_df = pd.read_csv('pcfg_programs.csv')
    
    # Count occurrences of each texture
    plain_count = programs_df['program'].str.count('plain').sum()
    stripes_count = programs_df['program'].str.count('stripes').sum()
    dots_count = programs_df['program'].str.count('dots').sum()
    
    # Calculate average log probability for programs containing each texture
    plain_probs = programs_df[programs_df['program'].str.contains('plain')]['log_probability'].mean()
    stripes_probs = programs_df[programs_df['program'].str.contains('stripes')]['log_probability'].mean()
    dots_probs = programs_df[programs_df['program'].str.contains('dots')]['log_probability'].mean()
    
    logger.info("\nFeature Distribution Analysis:")
    logger.info(f"Occurrences in programs:")
    logger.info(f"  plain: {plain_count}")
    logger.info(f"  stripes: {stripes_count}")
    logger.info(f"  dots: {dots_count}")
    logger.info(f"\nAverage log probabilities:")
    logger.info(f"  plain: {plain_probs:.4f}")
    logger.info(f"  stripes: {stripes_probs:.4f}")
    logger.info(f"  dots: {dots_probs:.4f}")

def main():
    # Add test execution before main logic
    if '--test' in sys.argv:
        test_program_applies()
        return
        
    programs_file = Path('pcfg_programs.csv')
    states_file = Path('state_probabilities.csv')
    
    # Generate or load programs
    if not programs_file.exists():
        logger.info("Generating programs...")
        program_log_probs = generate_all_programs(productions, max_depth=4)
        
        programs_df = pd.DataFrame({
            'program': [prog.strip('"') for prog in program_log_probs.keys()],
            'log_probability': [program_log_probs[prog] for prog in program_log_probs.keys()]
        }).sort_values('log_probability', ascending=False)
        
        programs_df.to_csv(programs_file, index=False)
    else:
        logger.info("Loading existing programs...")
        programs_df = pd.read_csv(programs_file)
    
    # Add after loading/generating programs
    analyze_feature_distribution()
    
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
        print(f"  Shapes: {row['shape1_sides']}/{row['shape1_shade']}/{row['shape1_texture']}, "
              f"{row['shape2_sides']}/{row['shape2_shade']}/{row['shape2_texture']}, "
              f"{row['shape3_sides']}/{row['shape3_shade']}/{row['shape3_texture']}"
              f" (log prob: {row['log_probability']:.2f})")
    
    logger.info("Done!")

if __name__ == "__main__":
    main()

