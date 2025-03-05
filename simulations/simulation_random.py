from itertools import product
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
from pathlib import Path
from rllib.shapeworld import State, Shape

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

def state_to_dict(state: State) -> dict:
    """Convert a State object to a dictionary for CSV output."""
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

def calculate_random_probabilities(states: list[State]) -> pd.DataFrame:
    """Calculate uniform probability for each state."""
    logger.info("Calculating random state probabilities...")
    
    # For uniform random policy, each state has equal probability
    num_states = len(states)
    log_probability = -np.log(num_states)  # Negative log likelihood for uniform distribution
    
    # Convert to DataFrame with flattened state features
    results = []
    for state in states:
        state_dict = state_to_dict(state)
        state_dict['log_probability'] = log_probability
        results.append(state_dict)
    
    states_df = pd.DataFrame(results)
    
    # Verify normalization
    total_prob = np.sum(np.exp(states_df['log_probability']))
    logger.info(f"Sum of probabilities: {total_prob:.10f}")
    
    return states_df

def main():
    output_file = Path('random_state_probabilities.csv')
    
    # Generate state space
    logger.info("Generating state space...")
    states = generate_state_space()
    logger.info(f"Generated {len(states)} states")
    
    # Calculate random probabilities
    states_df = calculate_random_probabilities(states)
    
    # Save results
    logger.info("Saving state probabilities...")
    states_df.to_csv(output_file, index=False)
    
    # Show sample of states with their probabilities
    logger.info("\nSample states (all should have same probability):")
    for _, row in states_df.head().iterrows():
        print(f"  Shapes: {row['shape1_sides']}/{row['shape1_shade']}/{row['shape1_texture']}, "
              f"{row['shape2_sides']}/{row['shape2_shade']}/{row['shape2_texture']}, "
              f"{row['shape3_sides']}/{row['shape3_shade']}/{row['shape3_texture']}"
              f" (log prob: {row['log_probability']:.2f})")
    
    logger.info("Done!")

if __name__ == "__main__":
    main()
